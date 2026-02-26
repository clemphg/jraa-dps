"""
UNet + Wavelet/Diffusion Utilities
==================================

Implementation of a UNet architecture with attention, timestep embeddings, 
residual blocks, and optional multi-device decoding, together with utility 
layers and helpers commonly used in diffusion models.

Notes
-----
- Supports 1D/2D/3D variants via `dims` (convolutions & pooling).
- `UNetModel.to([...])` allows distributing encoder/decoder across 2 devices.
- `EncoderUNetModel` provides a half-UNet encoder with configurable pooling.

Based on: https://github.com/pfriedri/wdm-3d
"""

from abc import abstractmethod
import math
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Core utils
# --------------------------------------------------------------------------- #

class SiLU(nn.Module):
    """SiLU activation (Swish). Keeps compatibility with older torch versions."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that computes in float32 for stability, then casts back."""

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create a 1D/2D/3D convolution module based on `dims`."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs) -> nn.Linear:
    """Create an `nn.Linear` module."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create a 1D/2D/3D average pooling module based on `dims`."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate: float = 0.99) -> None:
    """Exponential moving average update of target params from source params."""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module: nn.Module) -> nn.Module:
    """Set all parameters of a module to zero (in-place) and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """Scale all parameters of a module (in-place) and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: th.Tensor) -> th.Tensor:
    """Mean over all non-batch dimensions (kept for API parity)."""
    return tensor.mean(dim=list(range(2, len(tensor.shape))))


def normalization(channels: int, groups: int = 32) -> nn.Module:
    """Factory for GroupNorm32 with the given number of groups."""
    return GroupNorm32(groups, channels)


def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: int = 10000) -> th.Tensor:
    """Create sinusoidal timestep embeddings (cos/sin pairs)."""
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        emb = th.cat([emb, th.zeros_like(emb[:, :1])], dim=-1)
    return emb


def checkpoint(func, inputs, params, flag):
    """Optional gradient checkpointing to reduce memory at extra compute cost."""
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    """Custom checkpoint wrapper to avoid storing intermediate activations."""

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            outputs = ctx.run_function(*ctx.input_tensors)
        return outputs

    @staticmethod
    def backward(ctx, *grads):
        # Recreate graph
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            outputs = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            outputs, ctx.input_tensors + ctx.input_params, grads, allow_unused=True
        )
        # Cleanup
        del ctx.input_tensors
        del ctx.input_params
        del outputs
        return (None, None) + input_grads


# --------------------------------------------------------------------------- #
# Timestep-aware container building blocks
# --------------------------------------------------------------------------- #

class TimestepBlock(nn.Module):
    """Mixin interface: forward(x, emb) where `emb` is a timestep embedding."""

    @abstractmethod
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """Apply the module to `x` given timestep embeddings `emb`."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential that forwards `emb` to submodules that accept it."""

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


# --------------------------------------------------------------------------- #
# Up/Down sampling
# --------------------------------------------------------------------------- #

class Upsample(nn.Module):
    """Nearest-neighbor upsampling with optional convolution."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3 and self.resample_2d:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Stride-2 downsampling via conv or average pooling."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = (1, 2, 2) if dims == 3 and resample_2d else 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


# --------------------------------------------------------------------------- #
# Residual & Attention Blocks
# --------------------------------------------------------------------------- #

class ResBlock(TimestepBlock):
    """Residual block with (optional) up/downsample and FiLM conditioning."""

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        num_groups=32,
        resample_2d=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_groups = num_groups
        self.updown = up or down

        # In layers
        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        if up:
            self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
        elif down:
            self.h_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # Embedding transform
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # Out layers
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        # Skip path
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """Apply the block to features `x`, with conditioning `emb`."""
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            # kept for parity with original behavior
            print("You use scale-shift norm")
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions (N-D)."""

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"channels {channels} not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x_ = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_ + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """THOP helper: estimate ops for attention (kept for compatibility)."""
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """Legacy QKV attention (split heads after reshape)."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """Alternate QKV attention (chunk q/k/v first, then split heads)."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# --------------------------------------------------------------------------- #
# UNet family
# --------------------------------------------------------------------------- #

class UNetModel(nn.Module):
    """Full UNet with attention and timestep conditioning."""

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        num_groups=32,
        bottleneck_attention=True,
        resample_2d=True,
        additive_skips=False,
        decoder_device_thresh=0,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.num_groups = num_groups
        self.bottleneck_attention = bottleneck_attention
        self.devices = None
        self.decoder_device_thresh = decoder_device_thresh
        self.additive_skips = additive_skips

        # Timestep MLP
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Encoder / input blocks
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # Input levels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, resample_2d=resample_2d)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.input_block_chans_bk = input_block_chans[:]  # preserved

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups, resample_2d=resample_2d,
            ),
            *(
                [
                    AttentionBlock(
                        ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order,
                        num_groups=self.num_groups,
                    )
                ] if self.bottleneck_attention else []
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups, resample_2d=resample_2d,
            ),
        )
        self._feature_size += ch

        # Decoder / output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                mid_ch = (
                    model_channels * mult
                    if not self.additive_skips
                    else (input_block_chans[-1] if input_block_chans else model_channels)
                )
                layers = [
                    ResBlock(
                        ch + ich if not self.additive_skips else ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            mid_ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )
                ch = mid_ch
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            mid_ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                            up=True, num_groups=self.num_groups, resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Upsample(mid_ch, conv_resample, dims=dims, out_channels=out_ch, resample_2d=resample_2d)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                mid_ch = ch

        # Final conv head
        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    # NOTE: kept identical multi-device distribution behavior, including prints
    def to(self, *args, **kwargs):
        """
        Override to() to optionally distribute parts of the model to multiple devices.
        Pass a list/tuple of two devices to split decoder after `decoder_device_thresh`.
        """
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
            assert not kwargs and len(args) == 1
            self.devices = args[0]
            self.input_blocks.to(self.devices[0])
            self.time_embed.to(self.devices[0])
            self.middle_block.to(self.devices[0])
            for k, b in enumerate(self.output_blocks):
                (b.to(self.devices[0]) if k < self.decoder_device_thresh else b.to(self.devices[1]))
            self.out.to(self.devices[0])
            print(f"distributed UNet components to devices {self.devices}")
        else:
            super().to(*args, **kwargs)
            if self.devices is None:
                p = next(self.parameters())
                self.devices = [p.device, p.device]

    def forward(self, x: th.Tensor, timesteps: th.Tensor, y: th.Tensor | None = None) -> th.Tensor:
        """
        Apply the model to an input batch.

        Args:
            x: Input tensor [N, C, ...] on devices[0].
            timesteps: 1-D tensor of time steps on devices[0].
            y: Optional class labels [N] if class-conditional.

        Returns:
            Output tensor [N, C, ...] on devices[0].
        """
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y iff the model is class-conditional"
        assert x.device == self.devices[0], f"{x.device=} does not match {self.devices[0]=}"
        assert timesteps.device == self.devices[0], f"{timesteps.device=} does not match {self.devices[0]=}"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        self.hs_shapes = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            self.hs_shapes.append(h.shape)

        h = self.middle_block(h, emb)

        for k, module in enumerate(self.output_blocks):
            new_hs = hs.pop()
            if k == self.decoder_device_thresh:
                h = h.to(self.devices[1])
                emb = emb.to(self.devices[1])
            if k >= self.decoder_device_thresh:
                new_hs = new_hs.to(self.devices[1])

            h = (h + new_hs) / 2 if self.additive_skips else th.cat([h, new_hs], dim=1)
            h = module(h, emb)

        h = h.to(self.devices[0])
        return self.out(h)


class SuperResModel(UNetModel):
    """UNet variant for super-resolution; concatenates upsampled low-res input."""

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x: th.Tensor, timesteps: th.Tensor, low_res: th.Tensor | None = None, **kwargs) -> th.Tensor:
        _, _, new_h, new_w = x.shape
        upsampled = F.interpolate(low_res, (new_h, new_w), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """Half-UNet encoder with attention and timestep embedding."""

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        num_groups=32,
        resample_2d=True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.num_groups = num_groups

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Input stack
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups, resample_2d=resample_2d,
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order,
                num_groups=self.num_groups,
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups, resample_2d=resample_2d,
            ),
        )
        self._feature_size += ch

        # Pooling head(s)
        self.pool = pool
        spatial_dims = (2, 3, 4, 5)[:dims]
        self.gap = lambda x: x.mean(dim=spatial_dims)  # global average pooling
        self.cam_feature_maps = None  # kept as in original
        print('pool', pool)  # kept (behavior parity)

        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch, self.num_groups),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            # Intentionally left as pass (original behavior)
            pass
        elif pool == "spatial":
            print('spatial')  # kept
            self.out = nn.Linear(256, self.out_channels)
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048, self.num_groups),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def forward(self, x: th.Tensor, timesteps: th.Tensor) -> th.Tensor:
        """
        Apply the encoder to an input batch.

        Args:
            x: Input tensor [N, C, ...].
            timesteps: 1-D tensor of time steps.

        Returns:
            If pool == "adaptive": [N, K] features via GAP + 1x1 conv + flatten.
            If pool == "spatial" or "spatial_v2": [N, K] via specified heads.
            If pool == "attention": original 'pass' behavior is preserved.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)

        if self.pool.startswith("spatial"):
            self.cam_feature_maps = h
            h = self.gap(h)
            N = h.shape[0]
            h = h.reshape(N, -1)
            print('h1', h.shape)  # kept
            return self.out(h)
        else:
            h = h.type(x.dtype)
            self.cam_feature_maps = h
            return self.out(h)
