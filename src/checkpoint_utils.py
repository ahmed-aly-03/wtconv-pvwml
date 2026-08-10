"""Shared checkpoint-loading helper.

This exists because of a real incident: train_tdlf_finetune.py's Stage 1
saves its backbone under the key "backbone_state_dict", but
TDLFBackbone.load_pretrained only knew to unwrap "state_dict". The mismatch
meant the loader silently matched zero real parameter names (strict=False
swallowed it), leaving the "pretrained" backbone at its random init --
with only a quiet "missing keys: [...]" printed among thousands of lines
of training output. Centralizing the load logic here and hard-failing on a
near-total miss (rather than just printing) turns that class of bug into
an immediate crash instead of a silently-wrong training run.
"""
import torch


def load_state_dict_flexible(
    module,
    checkpoint_path: str,
    wrapper_keys=("state_dict",),
    drop_prefixes=("head.",),
    missing_ratio_threshold: float = 0.5,
    map_location="cpu",
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in wrapper_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    state_dict = {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in drop_prefixes)}
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    missing = [m for m in missing if not any(m.startswith(p) for p in drop_prefixes)]

    total = len(list(module.state_dict().keys()))
    if total > 0 and len(missing) / total > missing_ratio_threshold:
        raise RuntimeError(
            f"Refusing to continue: {len(missing)}/{total} parameter keys in "
            f"'{module.__class__.__name__}' were not found in checkpoint '{checkpoint_path}' "
            f"(tried wrapper keys {wrapper_keys}). This almost always means the checkpoint's "
            f"top-level structure doesn't match what this loader expects (e.g. a different "
            f"wrapper key), not that the backbone legitimately differs -- check how the "
            f"checkpoint was actually saved before retrying.\n"
            f"First few missing keys: {missing[:5]}"
        )

    print(f"Loaded checkpoint from {checkpoint_path}")
    if missing:
        shown = missing[:10]
        print(f"  missing keys ({len(missing)}): {shown}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        shown = unexpected[:10]
        print(f"  unexpected keys ({len(unexpected)}): {shown}{' ...' if len(unexpected) > 10 else ''}")
