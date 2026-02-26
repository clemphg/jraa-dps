"""
PET Projector (TOF and Non-TOF)
===============================

PET projector class that supports both Time-of-Flight (TOF) and non-TOF forward 
and back projections with optional subsets using the ParallelProj library:
https://parallelproj.readthedocs.io/en/stable/.

It also defines PyTorch-compatible autograd wrappers for integrating the projector
with deep learning workflows.
"""

import torch
import parallelproj
import array_api_compat.torch as xp
from copy import copy
from src.operators.operator import Operator


class Projector(Operator):
    """PET Projector supporting both TOF and Non-TOF modes."""

    def __init__(
        self,
        voxel_shape: tuple = (2, 1.953125, 1.953125),
        volume_shape: tuple = (64, 256, 256),
        num_rings: int = 24,
        max_ring_difference: int = 22,
        diameter: float = 700,
        num_sides: int = 300,
        num_lor_endpoints_per_side: int = 1,
        lor_spacing: int = 1,
        lor_radial_trim: int = 30,
        use_res_model: bool = True,
        fwhm: float = 4.5,
        use_tof: bool = True,
        num_tofbins: int = 13,
        tofbin_width: float = 60,
        sigma_tof: float = 24.50529087048832,
        num_sigmas: float = 2.0,
        tofcenter_offset: float = 0,
        num_subsets: int = None,
        device: str = "cuda",
    ) -> None:
        """Initializes the PET projector and its associated geometry.

        Args:
            voxel_shape (tuple): Dimensions of a voxel in mm (z, y, x).
            volume_shape (tuple): Dimensions of the 3D image volume (D, H, W).
            num_rings (int): Number of detector rings.
            max_ring_difference (int): Maximum ring difference allowed for LORs.
            diameter (float): Diameter of the detector ring in mm.
            num_sides (int): Number of sides of the regular polygonal scanner.
            num_lor_endpoints_per_side (int): Number of LOR endpoints per detector side.
            lor_spacing (int): Spacing between detectors on each side.
            lor_radial_trim (int): Number of detectors to trim radially.
            use_res_model (bool): Whether to apply a Gaussian resolution model.
            fwhm (float): Full width at half maximum (mm) of the Gaussian filter.
            use_tof (bool): Whether to use Time-of-Flight modeling.
            num_tofbins (int): Number of TOF bins.
            tofbin_width (float): Width of each TOF bin in mm.
            sigma_tof (float): TOF Gaussian sigma in mm.
            num_sigmas (float): Number of sigmas defining the TOF window.
            tofcenter_offset (float): Offset of the TOF center.
            num_subsets (int, optional): Number of subsets for OSEM reconstruction.
            device (str): Device for computation ('cuda' or 'cpu').
        """
        self.volume_shape = volume_shape
        self.num_subsets = num_subsets
        self.use_tof = use_tof
        self.device = device

        # --- Scanner Geometry Definition ---
        scanner = parallelproj.RegularPolygonPETScannerGeometry(
            xp,
            device,
            radius=diameter // 2,
            num_sides=num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            ring_positions=xp.linspace(
                -(volume_shape[0] // 2) * voxel_shape[2],
                (volume_shape[0] // 2) * voxel_shape[2],
                num_rings,
            ),
            symmetry_axis=0,  # For volume shape (D, H, W)
        )

        # --- LOR Descriptor ---
        lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=lor_radial_trim,
            max_ring_difference=max_ring_difference,
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
        )

        # --- Non-TOF and TOF Projectors ---
        proj = parallelproj.RegularPolygonPETProjector(
            lor_desc, img_shape=volume_shape, voxel_size=voxel_shape
        )
        proj_tof = parallelproj.RegularPolygonPETProjector(
            lor_desc, img_shape=volume_shape, voxel_size=voxel_shape
        )
        proj_tof.tof_parameters = parallelproj.TOFParameters(
            num_tofbins=num_tofbins,
            tofbin_width=tofbin_width,
            sigma_tof=sigma_tof,
            num_sigmas=num_sigmas,
            tofcenter_offset=tofcenter_offset,
        )

        # --- Resolution Model ---
        if use_res_model:
            self.res_model = parallelproj.GaussianFilterOperator(
                proj.in_shape, sigma=fwhm / (2.355 * proj.voxel_size)
            )
            self.projector = parallelproj.CompositeLinearOperator((proj, self.res_model))
            self.projector_tof = parallelproj.CompositeLinearOperator((proj_tof, self.res_model))
        else:
            self.projector = proj
            self.projector_tof = proj_tof

        # --- Subset Handling ---
        if num_subsets is not None:
            self._configure_subsets(proj, proj_tof, use_res_model, num_subsets)

        # --- Autograd-Compatible Operators ---
        self.project = LinearSingleChannelOperator.apply
        self.backproject = AdjointLinearSingleChannelOperator.apply

    def _configure_subsets(self, proj, proj_tof, use_res_model, num_subsets):
        """Configures subset projectors for ordered-subsets reconstruction."""
        subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
            num_subsets, len(proj.out_shape)
        )
        proj.clear_cached_lor_endpoints()

        subset_linops = []
        for i in range(num_subsets):
            subset_proj = copy(proj)
            subset_proj.views = subset_views[i]
            if use_res_model:
                subset_linops.append(
                    parallelproj.CompositeLinearOperator([subset_proj, self.res_model])
                )
            else:
                subset_linops.append(subset_proj)
        self.projector = parallelproj.LinearOperatorSequence(subset_linops)
        self.subset_views = subset_views
        self.subset_slices = subset_slices

        # Repeat for TOF
        subset_views_tof, subset_slices_tof = proj_tof.lor_descriptor.get_distributed_views_and_slices(
            num_subsets, len(proj_tof.out_shape)
        )
        proj_tof.clear_cached_lor_endpoints()

        subset_linops_tof = []
        for i in range(num_subsets):
            subset_proj_tof = copy(proj_tof)
            subset_proj_tof.views = subset_views_tof[i]
            if use_res_model:
                subset_linops_tof.append(
                    parallelproj.CompositeLinearOperator([subset_proj_tof, self.res_model])
                )
            else:
                subset_linops_tof.append(subset_proj_tof)
        self.projector_tof = parallelproj.LinearOperatorSequence(subset_linops_tof)
        self.subset_views_tof = subset_views_tof
        self.subset_slices_tof = subset_slices_tof

    def transform(self, x: torch.Tensor, subset_id: int = None, span: bool = False, use_tof: bool = None) -> torch.Tensor:
        """Forward projection.

        Args:
            x (torch.Tensor): Input image tensor.
            subset_id (int, optional): Index of subset to use (for OSEM).
            span (bool): Whether to apply span compression.
            use_tof (bool, optional): Override TOF flag.

        Returns:
            torch.Tensor: Forward-projected sinogram tensor.
        """
        is_tof = self.use_tof if use_tof is None else use_tof

        if subset_id is not None and span:
            raise ValueError("When using subsets, sinograms should not be spanned.")

        projector = (
            self.projector_tof[subset_id] if is_tof and subset_id is not None else
            self.projector_tof if is_tof else
            self.projector[subset_id] if subset_id is not None else
            self.projector
        )

        y = self.project(x, projector) if x.requires_grad else projector(x)

        if span:
            y = self.spanner.transform(y.unsqueeze(0)).squeeze(0)

        return y

    def transposed_transform(self, y: torch.Tensor, subset_id: int = None, span: bool = False, use_tof: bool = None) -> torch.Tensor:
        """Backprojection.

        Args:
            y (torch.Tensor): Input sinogram tensor.
            subset_id (int, optional): Index of subset to use (for OSEM).
            span (bool): Whether to unspan the sinogram.
            use_tof (bool, optional): Override TOF flag.

        Returns:
            torch.Tensor: Backprojected image tensor.
        """
        is_tof = self.use_tof if use_tof is None else use_tof

        if subset_id is not None and span:
            raise ValueError("When using subsets, sinograms should not be spanned.")

        projector = (
            self.projector_tof[subset_id] if is_tof and subset_id is not None else
            self.projector_tof if is_tof else
            self.projector[subset_id] if subset_id is not None else
            self.projector
        )

        y_unspanned = self.spanner.transposed_transform(y) if span else y

        x = self.backproject(y_unspanned, projector) if y.requires_grad else projector.adjoint(y_unspanned)
        return x



class LinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        """forward pass of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, num_voxels_x, num_voxels_y, num_voxels_z)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, opertor.out_shape)
        """

        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size,) + operator.out_shape, dtype=x.dtype, device=x.device
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, ...] = operator(x[i, 0, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, operator.out_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size, 1) + tuple(operator.in_shape),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, 0, ...] = operator.adjoint(grad_output[i, ...].detach())

            return x, None
        

class AdjointLinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing the adjoint of a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        """forward pass of the adjoint of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, operator.out_shape)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size, 1) + operator.in_shape, dtype=x.dtype, device=x.device
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, 0, ...] = operator.adjoint(x[i, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, 1, operator.in_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.out_shape)
        """

        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size,) + operator.out_shape,
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, ...] = operator(grad_output[i, 0, ...].detach())

            return x, None