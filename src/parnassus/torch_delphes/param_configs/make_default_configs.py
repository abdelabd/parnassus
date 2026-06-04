"""Regenerate the shipped param-config YAMLs.

Run with:  python -m parnassus.torch_delphes.param_configs.make_default_configs

Produces, next to this file:

- ``cms_target_default.yaml``: every parameter at its model default, with the
  three energy/momentum scales overridden to the reachable ground-truth values
  (CHAD 1.25, ECal 1.20, HCal 0.90). All ``trainable: false``. Use this as the
  pseudodata-generation truth.
- ``debug_train_chad_scale_barrel.yaml``: identical, except the charged-hadron
  barrel pT-scale element starts off-truth at 1.0 and is the *only* trainable
  parameter -- the minimal single-knob identifiability debug.

These are intentionally derived programmatically from the live model so they
always cover exactly the current parameter set; edit the YAMLs (or this script)
to change targets / trainable flags.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes import param_config as pc

_HERE = Path(__file__).resolve().parent

# Reachable (inside the open (0.7,1.3)) ground-truth scales for the target card.
_TARGET_SCALES = {
    "ChargedHadronMomentumSmearing.resolution_module.scale_raw": 1.25,
    "ECal.scale_module.scale_raw": 1.20,
    "HCal.scale_module.scale_raw": 0.90,
}
# The single parameter the debug config trains, and its off-truth start value.
_DEBUG_KEY = "ChargedHadronMomentumSmearing.resolution_module.scale_raw[0]"
_DEBUG_START = 1.0


def main() -> None:
    """Write the two shipped configs next to this module."""
    torch.manual_seed(0)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    named = dict(card.named_parameters())
    with torch.no_grad():
        for name, scale in _TARGET_SCALES.items():
            p = named[name]
            p.copy_(pc.to_raw(name, [scale] * p.numel()).reshape(p.shape))

    target_path = _HERE / "cms_target_default.yaml"
    pc.dump_param_config(card, target_path)
    print(f"wrote {target_path}")

    with open(target_path) as f:
        cfg = yaml.safe_load(f)
    cfg[_DEBUG_KEY]["value"] = _DEBUG_START
    cfg[_DEBUG_KEY]["trainable"] = True
    debug_path = _HERE / "debug_train_chad_scale_barrel.yaml"
    with open(debug_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {debug_path}")


if __name__ == "__main__":
    main()
