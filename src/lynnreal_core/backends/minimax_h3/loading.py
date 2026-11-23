"""Portable identity for a strictly loaded H3 inference model."""
from collections.abc import Mapping
from typing import Any
from lynnreal_core.errors import BackendContractError

def _base_model_profile(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Portable, readable identity for a compatible MiniMax-H3 base model."""

    profile = {
        "schema": "lynnreal_core.minimax-h3-base-profile.v1",
        "family": "minimax_h3",
        "architecture": str(model_cfg.get("architecture", "minimax-h3-33b")),
        "component": str(model_cfg.get("transformer_subfolder", "transformer")),
        "parameter_dtype_profile": "bfloat16-blocks+float32-io-timestep-v1",
    }
    revision = str(model_cfg.get("revision") or "").strip()
    if revision:
        profile["revision"] = revision
    return profile

def _base_model_load_receipt(model_cfg: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """Describe the strictly loaded model with portable semantic fields."""

    parameters = tuple(model.named_parameters())
    if not parameters or len({name for name, _ in parameters}) != len(parameters):
        raise BackendContractError("H3 base model parameter inventory is empty or duplicated")
    dtype_inventory: dict[str, dict[str, int]] = {}
    for _name, parameter in parameters:
        dtype = str(parameter.dtype).removeprefix("torch.")
        entry = dtype_inventory.setdefault(dtype, {"tensors": 0, "parameters": 0})
        entry["tensors"] += 1
        entry["parameters"] += int(parameter.numel())
    return {
        "schema": "lynnreal_core.minimax-h3-base-load.v1",
        "profile": _base_model_profile(model_cfg),
        "strict_state_load": {
            "missing_keys": 0,
            "unexpected_keys": 0,
            "mismatched_shapes": 0,
            "load_errors": 0,
        },
        "parameter_inventory": {
            "tensors": len(parameters),
            "parameters": sum(int(parameter.numel()) for _, parameter in parameters),
            "dtypes": dict(sorted(dtype_inventory.items())),
        },
    }
