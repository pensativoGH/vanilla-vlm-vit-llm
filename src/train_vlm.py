"""Training helpers for the VLM notebook."""

from __future__ import annotations

import os

import torch

from src.train_helper import validate_vlm


def save_checkpoint(
    ckpt_path: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loss: list,
    opt_cfg,
) -> None:
    """Save the model, optimizer, and scheduler state dictionaries."""
    raw_model = getattr(model, "_orig_mod", model) if opt_cfg.compile is not None else model
    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "train_loss": train_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    tmp = ckpt_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)


def train_vlm(
    model,
    train_loader,
    val_loader=None,
    optimizer=None,
    device=None,
    opt_cfg=None,
    scheduler=None,
    output_dir=None,
    max_steps=None,
    validation=True,
):
    """Train the VLM model."""
    model.train()

    step = 0
    train_loss: list[float] = []
    val_loss: list[float] = []
    print_every = max_steps // 20
    save_every = max_steps // 10

    while step < max_steps:
        for img, text_tokens, attention_mask, targets in train_loader:
            img = img.to(device)
            text_tokens = text_tokens.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if opt_cfg.autocast is not None:
                with torch.autocast(device_type="cuda", dtype=opt_cfg.autocast_dtype):
                    logits, loss = model(
                        x_img=img,
                        x_text=text_tokens,
                        targets=targets,
                        attention_mask=attention_mask,
                    )
            else:
                logits, loss = model(
                    x_img=img,
                    x_text=text_tokens,
                    targets=targets,
                    attention_mask=attention_mask,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if opt_cfg.scheduler:
                scheduler.step()

            train_loss.append(loss.item())
            step += 1

            if step % print_every == 0:
                print(f"Step {step} | Loss {loss.item():.4f}")

            if step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"vlm_coco_captions{step:07d}.pt")
                save_checkpoint(
                    ckpt_path=ckpt_path,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=train_loss,
                    opt_cfg=opt_cfg,
                )
                print(f"Checkpoint saved at step {step}")

                if val_loader is not None and validation:
                    model.eval()
                    with torch.inference_mode():
                        v_loss = validate_vlm(
                            model,
                            val_loader,
                            device,
                            max_batches=20,
                            autocast_dtype=opt_cfg.autocast_dtype,
                        )
                        print(f"Validation loss: {v_loss:.4f}")
                        val_loss.append(v_loss)
                    model.train()

            if step >= max_steps:
                break

    ckpt_path = os.path.join(output_dir, "vlm_coco_captions_rope_final.pt")
    torch.save(
        {"step": step, "model": model.state_dict(), "train_loss": train_loss},
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    return train_loss, val_loss
