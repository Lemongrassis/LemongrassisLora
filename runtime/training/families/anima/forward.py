"""Anima family training forward pass (multi-model PR-2b, moved from training/model_loading.py).

forward_with_optional_checkpoint manually unrolls Anima's internal API
(prepare_embedded_sequence / t_embedder / blocks / final_layer / unpatchify)
to do per-block gradient checkpointing -- pure family knowledge (03 S4.3:
the checkpoint unrolling strategy is model-private).
"""

from __future__ import annotations


def partial_checkpoint_blocks(total: int, skip_pct: int, swapped: int = 0) -> int:
    """How many leading blocks run without checkpointing under partial checkpointing.

    Only the leading blocks are exempt: block swap offloads the trailing
    ``swapped`` blocks, and those must stay checkpointed -- without the
    recompute, autograd would keep references to the shared GPU slot buffers
    that the next swapped block overwrites.
    """
    skip = total * max(0, min(100, int(skip_pct))) // 100
    return max(0, min(skip, total - max(0, int(swapped))))


def forward_with_optional_checkpoint(model, latents, timesteps, cross, padding_mask, use_checkpoint=False):
    """Forward pass with optional gradient checkpointing.

    ``model.checkpoint_skip_blocks`` (set by partial checkpointing) leaves that
    many leading blocks un-checkpointed.
    """
    if not use_checkpoint:
        return model(latents, timesteps, cross, padding_mask=padding_mask)
    from torch.utils.checkpoint import checkpoint

    x_B_T_H_W_D, rope_emb, extra_pos_emb = model.prepare_embedded_sequence(
        latents, fps=None, padding_mask=padding_mask,
    )
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = model.t_embedder(timesteps)
    t_embedding = model.t_embedding_norm(t_embedding)

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb,
        "adaln_lora_B_T_3D": adaln_lora,
        "extra_per_block_pos_emb": extra_pos_emb,
    }

    skip = int(getattr(model, "checkpoint_skip_blocks", 0) or 0)
    for i, block in enumerate(model.blocks):
        def custom_forward(x, blk=block):
            return blk(x, t_embedding, cross, **block_kwargs)
        if i < skip:
            x_B_T_H_W_D = custom_forward(x_B_T_H_W_D)
        else:
            x_B_T_H_W_D = checkpoint(custom_forward, x_B_T_H_W_D, use_reentrant=False)

    x_B_T_H_W_O = model.final_layer(x_B_T_H_W_D, t_embedding, adaln_lora_B_T_3D=adaln_lora)
    return model.unpatchify(x_B_T_H_W_O)
