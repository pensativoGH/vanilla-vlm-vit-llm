from __future__ import annotations

import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import SiglipVisionModel

from src.configs import ConfigParametersLLM, ConfigParametersVLM, ConfigParametersViT
from src.model.gpt import GPT, TransformerConfig
from src.model.vlm import CustomViTAdapter, SigLIPAdapter, VLM
from src.model.vit import ViT


def vlm_from_config(config: dict) -> VLM:
    llm = llm_from_config(config["llm"], device=config["device"])
    vision_encoder = vision_encoder_from_config(config["vision_encoder"], device=config["device"])

    vlm_cfg = ConfigParametersVLM(
        model_dim=llm.model_dim,
        vision_model_dim=vision_encoder.model_dim,
        device=config["device"],
        vision_encoder=vision_encoder,
        LLM=llm,
        batch_size=config["vlm"]["batch_size"],
    )
    return VLM(vlm_cfg)


def llm_from_config(llm_cfg: dict, tfr_cfg: dict, device) -> nn.Module:

    if llm_cfg["type"] == "custom_gpt":
        with open(llm_cfg["config_path"], "r") as f:
            params = json.load(f)

        params["device"] = device
        cfg = ConfigParametersLLM.from_dict(params)

        with open(tfr_cfg["config_path"], "r") as f:
            params = json.load(f)

        cfg_tfr = TransformerConfig.from_dict(params)

        model = GPT(cfg, cfg_tfr)

        if "checkpoint_path" in llm_cfg:
            payload = torch.load(llm_cfg["checkpoint_path"], map_location=device)
            state_dict = payload["model"] if "model" in payload else payload
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        return model

    raise ValueError(f"Unknown LLM type: {llm_cfg['type']}")


def vision_encoder_from_config(vision_cfg: dict, device) -> nn.Module:
    if vision_cfg["type"] == "custom_vit":
        with open(vision_cfg["config_path"], "r") as f:
            params = json.load(f)

        params["device"] = device
        cfg = ConfigParametersViT(**params)
        vit = ViT(cfg)

        if "checkpoint_path" in vision_cfg:
            payload = torch.load(vision_cfg["checkpoint_path"], map_location=device)
            state_dict = payload["model"] if "model" in payload else payload
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            vit.load_state_dict(state_dict)

        return CustomViTAdapter(vit)

    if vision_cfg["type"] == "siglip":
        model_name = vision_cfg["model_name"]
        siglip = SiglipVisionModel.from_pretrained(model_name)

        if vision_cfg.get("freeze", False):
            for param in siglip.parameters():
                param.requires_grad = False

        siglip = siglip.to(device)

        if "checkpoint_path" in vision_cfg:
            payload = torch.load(vision_cfg["checkpoint_path"], map_location=device)
            state_dict = payload["model"] if "model" in payload else payload
            siglip.load_state_dict(state_dict)

        return SigLIPAdapter(siglip)

    raise ValueError(f"Unknown vision encoder type: {vision_cfg['type']}")


def draw_plot(loss_list):
    window = 100
    if len(loss_list) >= window:
        smoothed = [
            sum(loss_list[i : i + window]) / window
            for i in range(len(loss_list) - window + 1)
        ]
        plt.plot(loss_list, alpha=0.2, label="raw")
        plt.plot(range(window - 1, len(loss_list)), smoothed, label="smoothed")
    else:
        plt.plot(loss_list, label="raw")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("VLM Train Loss")
    plt.legend()
    plt.show()
