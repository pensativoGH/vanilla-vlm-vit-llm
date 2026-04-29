"""Vision language model that pairs the custom ViT/Siglip with the custom GPT."""



from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.model.gpt import Linear
from src.configs import ConfigParametersVLM


class CustomViTAdapter(nn.Module):
      def __init__(self, vit: nn.Module) -> None:
          super().__init__()
          self.vit = vit
          self.model_dim = vit.model_dim
    
      def get_image_tokens(self, x: Tensor) -> Tensor:
          # returns [B, N, D] without CLS token
          return self.vit.encode(x)

      def forward(self, x: Tensor) -> Tensor:
          return self.get_image_tokens(x)


class SigLIPAdapter(nn.Module):
      def __init__(self, siglip_vision_model: nn.Module) -> None:
          super().__init__()
          self.model = siglip_vision_model
          self.model_dim = self.model.config.hidden_size

          for param in self.model.parameters():
              param.requires_grad = False
         

      def get_image_tokens(self, x: Tensor) -> Tensor:
          outputs = self.model(pixel_values=x)
          tokens = outputs.last_hidden_state   # [B, N, D]
          return tokens

      def forward(self, x: Tensor) -> Tensor:
          return self.get_image_tokens(x)


class VLM(nn.Module):
    """Vision language model: frozen ViT -> projector -> GPT."""

    def __init__(self, cfg: ConfigParametersVLM) -> None:
        super().__init__()
     
        self.vision_encoder = cfg.vision_encoder
        self.LLM = cfg.LLM
        self.model_dim = cfg.model_dim
        self.vision_model_dim = cfg.vision_model_dim
        self.vision_proj = Linear(cfg.vision_model_dim, cfg.model_dim)
        self.pos_emb_type = cfg.pos_emb_type


    def forward(
        self,
        x_img: Tensor,
        x_text: Tensor,
        targets: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | float]:
        """Run image + text through the VLM and optionally compute loss."""
        img_logits = self.vision_encoder(x_img)
        img_embeddings = self.vision_proj(img_logits)
        text_embeddings = self.LLM.token_embeddings(x_text)
        llm_embeddings = torch.cat([img_embeddings, text_embeddings], dim=1)

        #add position embeddings if absolute position embedding type is provided, skip if rope is used
        if self.pos_emb_type is None or self.pos_emb_type == "absolute":
            pos_embeddings = self.LLM.pos_emb(
                torch.arange(llm_embeddings.shape[1], device=llm_embeddings.device)
            )
            llm_embeddings = llm_embeddings + pos_embeddings

        B, N, _ = img_embeddings.shape

        # Mask out image positions in the targets so loss is only on text tokens
        if targets is not None:
            pad = torch.full((B, N), -100, dtype=targets.dtype, device=targets.device)
            targets = torch.cat([pad, targets], dim=1)

        # Extend attention mask to cover image tokens (always visible)
        if attention_mask is not None:
            img_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        return self.LLM(
            input_embeds=llm_embeddings,
            targets=targets,
            attention_mask=attention_mask,
            causal=True,
        )

    @torch.no_grad()
    def generate(
        self,
        x_img: Tensor,
        x_text: Tensor,
        attention_mask: Tensor | None,
        max_new_tokens: int,
        eos_token_id: int,
    ) -> Tensor:
        """Greedy decoding given an image and a text prompt."""
        img_embeddings = self.vision_proj(self.vision_encoder(x_img))
        text_embeddings = self.LLM.token_embeddings(x_text)
        llm_embeddings = torch.cat([img_embeddings, text_embeddings], dim=1)

        B, N, _ = img_embeddings.shape
        device = llm_embeddings.device

        if self.pos_emb_type is None or self.pos_emb_type == "absolute":
            pos_ids = torch.arange(llm_embeddings.shape[1], device=device)
            llm_embeddings = llm_embeddings + self.LLM.pos_emb(pos_ids)

        if attention_mask is not None:
            img_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        generated_ids_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits, _ = self.LLM(
                input_embeds=llm_embeddings,
                targets=None,
                attention_mask=attention_mask,
                causal=True,
            )

            T_cur = llm_embeddings.shape[1]
            logits = logits.view(B, T_cur, self.LLM.vocab_size)

            next_logits = logits[:, -1, :]
            next_ids = torch.argmax(next_logits, dim=-1, keepdim=True)

            next_ids = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_ids, eos_token_id),
                next_ids,
            )
            generated_ids_list.append(next_ids)

            finished = finished | (next_ids.squeeze(1) == eos_token_id)
            if finished.all():
                break

            
            next_emb = self.LLM.token_embeddings(next_ids)

            #add position embeddings if absolute position embedding type is provided, skip if rope is used
            if self.pos_emb_type is None or self.pos_emb_type == "absolute":
                next_pos = torch.tensor([llm_embeddings.shape[1]], device=device)
                next_emb = next_emb + self.LLM.pos_emb(next_pos)

            llm_embeddings = torch.cat([llm_embeddings, next_emb], dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype, device=device)],
                    dim=1,
                )

        return torch.cat(generated_ids_list, dim=1)
