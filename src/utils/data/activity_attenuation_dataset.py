import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Optional augmentations (enable only if torchio is available)
try:
    from torchio import RandomFlip, RandomAffine
    _HAS_TORCHIO = True
except ImportError:
    _HAS_TORCHIO = False


class ActivityAttenuationDataset(Dataset):
    """CHU Poitiers PET/CT Dataset (3D patches)."""

    def __init__(
        self,
        path: str,
        id_patients: list,
        nb_slices: int = 16,
        to_tensor: bool = True,
        norm_method_act=None,
        norm_method_atn=None,
        use_act: bool = True,
        use_atn: bool = True,
        clip_act: float | None = None,
        clip_atn: float | None = None,
        augment_flip: bool = False,
        augment_affine: bool = False,
        return_fn: bool = False,
        patch_suffix: str | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            path: Directory containing `.npy` patch files.
            id_patients: List of patient identifiers to include.
            nb_slices: Number of axial slices to extract per sample.
            to_tensor: Convert numpy arrays to torch tensors.
            norm_method_act: Normalization method for activity maps.
            norm_method_atn: Normalization method for attenuation maps.
            use_act: Include PET activity volumes.
            use_atn: Include attenuation volumes.
            clip_act: Max intensity for PET activity (if not None).
            clip_atn: Max intensity for attenuation map (if not None).
            augment_flip: Apply random flips (requires torchio).
            augment_affine: Apply random affine transforms (requires torchio).
            return_fn: Whether to return filenames alongside data.
            patch_suffix: Restrict to patches with a given filename suffix.
        """
        super().__init__()
        self.path = path
        self.id_patients = [str(pid) for pid in id_patients]
        self.nb_slices = nb_slices
        self.to_tensor = to_tensor
        self.norm_method_act = norm_method_act
        self.norm_method_atn = norm_method_atn
        self.clip_act = clip_act
        self.clip_atn = clip_atn
        self.augment_flip = augment_flip and _HAS_TORCHIO
        self.augment_affine = augment_affine and _HAS_TORCHIO
        self.return_fn = return_fn

        # Gather valid patch files
        all_files = [f for f in os.listdir(path) if f.endswith(".npy")]
        patches = [
            f for f in all_files
            if f.split("_")[0] in self.id_patients
        ]
        if patch_suffix:
            patches = [f for f in patches if f.split(".")[0].endswith(patch_suffix)]

        if not patches:
            raise FileNotFoundError(f"No matching .npy files found in {path} for given IDs.")

        self.patches_filenames = sorted(patches)
        self.num_channels = int(use_act) + int(use_atn)

    def __len__(self) -> int:
        """Return the number of available patient patches."""
        return len(self.patches_filenames)

    def __getitem__(self, idx: int):
        """Load and preprocess a single sample."""
        filename = os.path.join(self.path, self.patches_filenames[idx])
        image = np.load(filename).astype(np.float32)

        # Clip volume depth if needed
        depth = image.shape[1]
        if self.nb_slices > depth:
            raise ValueError(
                f"Requested {self.nb_slices} slices, but volume has only {depth}."
            )
        image = image[:, -self.nb_slices:] # Take the last nb_slices along axial axis

        # Convert to tensor
        if self.to_tensor:
            image = torch.from_numpy(image)

        # Convert to [C, D, H, W]
        if image.ndim != 4:
            image = image.unsqueeze(0)

        # Values clipping
        image[0] = torch.clamp(image[0], max=self.clip_act)
        image[1] = torch.clamp(image[1], max=self.clip_atn)

        # Normalization
        if self.norm_method_act:
            image[0] = self.norm_method_act.norm(image[0])
        if self.norm_method_atn:
            image[1] = self.norm_method_atn.norm(image[1])

        # Augmentation (if available)
        if _HAS_TORCHIO and (self.augment_flip or self.augment_affine):
            if self.augment_flip:
                image = RandomFlip(axes=(0, 1, 2), flip_probability=0.5)(image)
            if self.augment_affine:
                image = RandomAffine(
                    scales=(1, 1, 0.8, 1.1, 0.8, 1.1),
                    degrees=(-15, 15, 0, 0, 0, 0),
                    translation=0,
                    isotropic=False,
                    center="image",
                    image_interpolation="linear",
                    check_shape=True,
                )(image)

        if self.return_fn:
            return self.patches_filenames[idx], image
        return image
