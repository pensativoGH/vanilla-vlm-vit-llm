from __future__ import annotations

import glob
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import SiglipVisionModel

from src.configs import ConfigParametersLLM, ConfigParametersVLM, ConfigParametersViT, TransformerBlockConfig
from src.model.gpt import GPT
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

        cfg_tfr = [TransformerBlockConfig.from_dict(params) for _ in range(cfg.num_blocks)]

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


def count_module_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def summary_run_spec(
    run_name: str,
    llm_config: dict[str, Any],
    transformer_block_config: dict[str, Any],
) -> tuple[ConfigParametersLLM, list[TransformerBlockConfig]]:
    run_overrides = {
        "position_embedding": {
            "llm": {
                "pos_emb_type": "absolute",
            },
            "block": {
                "attention_type": "mha",
                "pos_emb_type": "absolute",
            },
        },
        "rope": {
            "llm": {
                "output_dir": "../outputs/llm_rope",
                "pos_emb_type": "rope",
            },
            "block": {
                "attention_type": "mha",
                "pos_emb_type": "rope",
            },
        },
        "gqa_rope": {
            "llm": {
                "output_dir": "../outputs/llm_rope_gqa",
                "pos_emb_type": "rope",
            },
            "block": {
                "attention_type": "gqa",
                "pos_emb_type": "rope",
            },
        },
    }

    if run_name not in run_overrides:
        raise ValueError(f"Unknown run name: {run_name}")

    llm_dict = dict(llm_config)
    llm_dict.update(run_overrides[run_name]["llm"])
    llm_cfg = ConfigParametersLLM.from_dict(llm_dict)

    block_dict = dict(transformer_block_config)
    block_dict.update(run_overrides[run_name]["block"])
    transformer_cfgs = [TransformerBlockConfig.from_dict(block_dict) for _ in range(llm_cfg.num_blocks)]
    return llm_cfg, transformer_cfgs


def training_seconds_from_checkpoints(output_dir: str) -> tuple[float | None, list[str]]:
    checkpoint_paths = sorted(glob.glob(os.path.join(output_dir, "llm_wikitext_step*.pt")))
    if len(checkpoint_paths) < 2:
        return None, checkpoint_paths

    checkpoint_times = [os.path.getmtime(path) for path in checkpoint_paths]
    return max(checkpoint_times) - min(checkpoint_times), checkpoint_paths


def checkpoint_path_for_summary(output_dir: str) -> str:
    final_checkpoint_names = [
        "llm_wikitext_rope_final.pt",
        "llm_wikitext_final.pt",
        "llm_final.pt",
    ]
    for checkpoint_name in final_checkpoint_names:
        final_checkpoint = os.path.join(output_dir, checkpoint_name)
        if os.path.exists(final_checkpoint):
            return final_checkpoint

    checkpoint_paths = sorted(glob.glob(os.path.join(output_dir, "llm_wikitext_step*.pt")))
    if checkpoint_paths:
        return checkpoint_paths[-1]

    raise FileNotFoundError(f"No GPT checkpoint found in {output_dir}")


def normalize_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("_orig_mod.")
        normalized_key = normalized_key.replace(".MHSA.", ".attention.")
        normalized_key = normalized_key.replace(".FFN.", ".mlp.")
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def infer_llm_config_from_checkpoint(
    llm_cfg: ConfigParametersLLM,
    normalized_state_dict: dict[str, torch.Tensor],
) -> ConfigParametersLLM:
    llm_dict = dict(llm_cfg.__dict__)
    if "logit_proj.W" in normalized_state_dict:
        llm_dict["logit_projection_type"] = "custom_linear"
    elif "logit_proj.weight" in normalized_state_dict:
        llm_dict["logit_projection_type"] = "linear"
    return ConfigParametersLLM.from_dict(llm_dict)


def infer_transformer_configs_from_checkpoint(
    transformer_cfgs: list[TransformerBlockConfig],
    normalized_state_dict: dict[str, torch.Tensor],
) -> list[TransformerBlockConfig]:
    inferred_cfgs = []
    for block_index, block_cfg in enumerate(transformer_cfgs):
        block_dict = dict(block_cfg.__dict__)
        q_proj_key = f"blocks.{block_index}.attention.q_proj.W"
        k_proj_key = f"blocks.{block_index}.attention.k_proj.W"

        if q_proj_key in normalized_state_dict and k_proj_key in normalized_state_dict:
            q_proj_shape = normalized_state_dict[q_proj_key].shape
            k_proj_shape = normalized_state_dict[k_proj_key].shape
            model_dim = q_proj_shape[0]
            q_out_dim = q_proj_shape[1]
            k_out_dim = k_proj_shape[1]

            block_dict["model_dim"] = model_dim

            if q_out_dim == k_out_dim:
                block_dict["attention_type"] = "mha"
            else:
                block_dict["attention_type"] = "gqa"
                q_heads = block_dict["q_heads"]
                head_dim = model_dim // q_heads
                block_dict["kv_heads"] = k_out_dim // head_dim

        inferred_cfgs.append(TransformerBlockConfig.from_dict(block_dict))

    return inferred_cfgs


def load_gpt_from_output_dir(
    llm_cfg: ConfigParametersLLM,
    transformer_cfgs: list[TransformerBlockConfig],
    output_dir: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    checkpoint_path = checkpoint_path_for_summary(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    normalized_state_dict = normalize_checkpoint_state_dict(checkpoint["model"])
    inferred_llm_cfg = infer_llm_config_from_checkpoint(llm_cfg, normalized_state_dict)
    inferred_transformer_cfgs = infer_transformer_configs_from_checkpoint(transformer_cfgs, normalized_state_dict)
    loaded_model = GPT(inferred_llm_cfg, inferred_transformer_cfgs).to(device)
    loaded_model.load_state_dict(normalized_state_dict)
    loaded_model.eval()
    return loaded_model, checkpoint, checkpoint_path


def print_gpt_run_summary(
    run_name: str,
    output_dir: str,
    device: torch.device,
    llm_config: dict[str, Any],
    transformer_block_config: dict[str, Any],
) -> None:
    llm_cfg, transformer_cfgs = summary_run_spec(run_name, llm_config, transformer_block_config)
    loaded_model, checkpoint, loaded_checkpoint_path = load_gpt_from_output_dir(
        llm_cfg=llm_cfg,
        transformer_cfgs=transformer_cfgs,
        output_dir=output_dir,
        device=device,
    )

    total_params = count_module_parameters(loaded_model)
    pos_emb_params = count_module_parameters(loaded_model.pos_emb) if hasattr(loaded_model, "pos_emb") else 0
    elapsed_seconds, checkpoint_paths = training_seconds_from_checkpoints(output_dir)

    print(f"Run: {run_name}")
    print(f"Output dir: {output_dir}")
    print(f"Loaded checkpoint: {os.path.basename(loaded_checkpoint_path)}")
    if "step" in checkpoint:
        print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Position embedding parameters: {pos_emb_params:,}")

    if elapsed_seconds is None:
        print("Training time from checkpoints: unavailable (need at least 2 step checkpoints)")
    else:
        print(f"Training time from checkpoints: {elapsed_seconds / 60:.2f} minutes ({elapsed_seconds:.1f} seconds)")
        print(f"Checkpoint range: {os.path.basename(checkpoint_paths[0])} -> {os.path.basename(checkpoint_paths[-1])}")

    for block_index, block in enumerate(loaded_model.blocks):
        attention = block.attention
        attention_params = count_module_parameters(attention)
        q_proj_params = count_module_parameters(attention.q_proj)
        k_proj_params = count_module_parameters(attention.k_proj)
        v_proj_params = count_module_parameters(attention.v_proj)
        o_proj_params = count_module_parameters(attention.o_proj)

        if hasattr(attention, "num_heads"):
            query_head_count = attention.num_heads
            head_description = f"num_heads={attention.num_heads}"
        else:
            query_head_count = attention.q_heads
            head_description = f"q_heads={attention.q_heads}, kv_heads={attention.kv_heads}"

        params_per_query_head = attention_params // query_head_count
        print(
            f"Block {block_index}: {attention.__class__.__name__}, attention_params={attention_params:,}, "
            f"{head_description}, params_per_query_head={params_per_query_head:,}"
        )
        print(
            f"  q_proj={q_proj_params:,}, k_proj={k_proj_params:,}, "
            f"v_proj={v_proj_params:,}, o_proj={o_proj_params:,}"
        )
