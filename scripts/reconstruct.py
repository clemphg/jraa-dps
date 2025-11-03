"""
inference.py
============

Configurable inference script for Wavelet Diffusion PET reconstruction.

Reads:
    - data.yml   (data and normalization)
    - model.yml        (model architecture)
    - projector.yml    (projector parameters)
    - inference.yml    (sampling parameters and environment)

Usage:
    python reconstruct.py \
        --data configs/data.yml \
        --model configs/model.yml \
        --proj configs/projector.yml \
        --infer configs/inference.yml
"""

import os
import sys
import torch
import argparse
import pickle as pkl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.train_infer import load_yaml, save_reconstruction
from src.utils.data.normalization import NormRange, NormSqrt
from src.utils.data.activity_attenuation_dataset import ActivityAttenuationDataset
from src.operators.projector import Projector
from src.operators.wavelet import Wavelet
from src.models.unet_wavelet_diffusion import UNetModel
from src.samplers.jraa_dps import JRAADPS


def main(args):
    # ---------------- LOAD CONFIGS ---------------- #
    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model)
    proj_cfg = load_yaml(args.proj)
    infer_cfg = load_yaml(args.infer)

    print(data_cfg)

    device = torch.device(infer_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nUsing device: {device}")

    # ---------------- DATASET ---------------- #
    with open(data_cfg["id_patients_path"], "rb") as f:
        id_patients = pkl.load(f)
    id_test = id_patients["test"]

    norm_act = NormSqrt(**data_cfg["norm_act"])
    norm_atn = NormRange(**data_cfg["norm_atn"])

    dataset = ActivityAttenuationDataset(
        path=data_cfg["path_data"],
        id_patients=id_test,
        nb_slices=data_cfg["nb_slices"],
        clip_act=data_cfg["clip_act"],
        clip_atn=data_cfg["clip_atn"],
    )
    print(f"Loaded {len(dataset)} measurement samples.\n")

    # ---------------- MODEL ---------------- #
    model_params = model_cfg["params"]
    model = UNetModel(**model_params)
    model.to(device)

    weights_path = os.path.expandvars(infer_cfg["weights_path"])
    print(f"Loading weights from: {weights_path}\n")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ---------------- OPERATORS ---------------- #
    projector = Projector(**proj_cfg["params"], num_subsets=infer_cfg["num_subsets"])
    projector_nosub = Projector(**proj_cfg["params"], num_subsets=None) # without subsets for simulating data
    wavelet = Wavelet()
    sampler_cfg = infer_cfg["sampler"]
    sampler = JRAADPS(model, device=device)

    # ---------------- SIMULATE DATA ---------------- #
    tau = proj_cfg["tau"]
    use_tof = proj_cfg["use_tof"]

    pet_img, mu_img = dataset[0]

    att_sino = torch.exp(-projector_nosub.transform(mu_img.to(device), use_tof=False))
    if use_tof:
        proj_act = projector_nosub.transform(pet_img.to(device), use_tof=True)
        att_sino = att_sino.unsqueeze(-1).repeat(1, 1, 1, proj_act.shape[-1])
    else:
        proj_act = projector_nosub.transform(pet_img.to(device), use_tof=False)

    # measurements
    ybar = torch.poisson(torch.clamp(tau * (att_sino * proj_act), min=0))

    print(f"Emission data simulated with tau={tau}, use_tof={use_tof} (shape: {ybar.shape})\n")


    # ---------------- GRADIENT FUNCTION ---------------- #
    def grad_fun(img_lambda, img_mu, y, subset_id, scatter=None):
        subset_slices = projector.subset_slices_tof if use_tof else projector.subset_slices

        img_lambda = torch.clamp(norm_act.denorm(img_lambda), 0, data_cfg["clip_act"])
        img_mu = torch.clamp(norm_atn.denorm(img_mu), 0, data_cfg["clip_atn"])

        att_sino = torch.exp(-projector.transform(img_mu, subset_id, use_tof=False))
        if use_tof:
            proj_act = projector.transform(img_lambda, subset_id, use_tof=True)
            att_sino = att_sino.unsqueeze(-1).repeat(1, 1, 1, 1, proj_act.shape[-1])
        else:
            proj_act = projector.transform(img_lambda, subset_id, use_tof=False)
        ybar = tau * (att_sino * proj_act)
        if scatter is not None:
            ybar = ybar + scatter[subset_slices[subset_id]].unsqueeze(0)

        loss = torch.nn.functional.poisson_nll_loss(
            ybar,
            y.squeeze()[subset_slices[subset_id]].unsqueeze(0).to(device),
            log_input=False,
        )
        return loss

    # ---------------- RECONSTRUCTION ---------------- #
    ybar = ybar.to(device)

    print("\n" + "-" * 25 + " RECONSTRUCTION " + "-" * 25 + "\n")

    z0 = sampler.inference(
        zT=torch.randn((1, 2*8, data_cfg["nb_slices"] // 2, data_cfg["img_dim"] // 2, data_cfg["img_dim"] // 2), device=device),
        grad_fun=lambda lbd, mu, subset_id, scatter: grad_fun(lbd, mu, ybar, subset_id, scatter),
        zeta=sampler_cfg["zeta"],
        xi=sampler_cfg["xi"],
        norm_grad=sampler_cfg["norm_grad"],
        num_subsets=infer_cfg["num_subsets"],
        display_progress=infer_cfg["display_progress"],
    )

    x0_res = wavelet.transposed_transform(z0, 3).cpu().squeeze()
    pet_recon = torch.clamp(norm_act.denorm(x0_res[0]), min=0).numpy()
    mu_recon = torch.clamp(norm_atn.denorm(x0_res[1]), min=0).numpy()

    save_reconstruction("jraa-dps-reconstruction", "./results/", np.stack([pet_recon, mu_recon], axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct with JRAA-DPS.")
    parser.add_argument("--data", 
                        default="configs/data.yaml",
                        help="Path to data.yml")
    parser.add_argument("--model", 
                        default="configs/model.yaml",
                        help="Path to model.yml")
    parser.add_argument("--proj", 
                        default="configs/projector.yaml",
                        help="Path to projector.yml")
    parser.add_argument("--infer", 
                        default="configs/inference.yaml",
                        help="Path to inference.yml")
    args = parser.parse_args()
    main(args)
