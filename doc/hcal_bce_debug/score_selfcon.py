"""Score the self-consistent-threshold stage-3 closures against the Phase-2b arms."""
import math
import sys
import yaml
import numpy as np

REPO = "/pscratch/sd/a/aelabd/parnassus"
truth = yaml.safe_load(open(f"{REPO}/src/parnassus/torch_delphes/param_configs/param_config_all.yaml"))
fits = {
    "ii-a (sampled_sigma)": "doc/figure_seq_hung_nBCE_condii_detach/fitted_config.yaml",
    "ii+selfcon": "doc/figure_seq_hung_nBCE_condii_selfcon/fitted_config.yaml",
    "iii+selfcon": "doc/figure_seq_hung_nBCE_condiii_selfcon/fitted_config.yaml",
}
loaded = {}
for k, v in fits.items():
    try:
        loaded[k] = yaml.safe_load(open(f"{REPO}/{v}"))
    except FileNotFoundError:
        print(f"(missing: {v})")

get = lambda d, k: d[k]["value"] if isinstance(d.get(k), dict) else d.get(k)
blocks = {
    "ECal_scales": [f"ECal.scale_module.scale_raw[{i}]" for i in range(3)],
    "ECal_res": [k for k in truth if "ECal.resolution_func" in k],
    "HCal_scales": [f"HCal.scale_module.scale_raw[{i}]" for i in range(2)],
    "HCal_res": [k for k in truth if "HCal.resolution_func" in k],
}
print("block".ljust(14) + "".join(f"{n:>22}" for n in loaded))
for name, keys in blocks.items():
    row = name.ljust(14)
    for d in loaded.values():
        errs = [abs(get(d, k) - truth[k]["value"]) / max(abs(truth[k]["value"]), 1e-9)
                for k in keys if get(d, k) is not None]
        row += f"{np.median(errs):>22.4f}"
    print(row)
print("\nHCal scales (config values are PHYSICAL; truth 0.8487 / 0.8934):")
for k, d in loaded.items():
    v = [get(d, f"HCal.scale_module.scale_raw[{i}]") for i in range(2)]
    print(f"  {k}: {v[0]:.4f} / {v[1]:.4f}")
print("ECal scales (truth "
      + " / ".join(f"{truth[f'ECal.scale_module.scale_raw[{i}]']['value']:.4f}" for i in range(3)) + "):")
for k, d in loaded.items():
    v = [get(d, f"ECal.scale_module.scale_raw[{i}]") for i in range(3)]
    print(f"  {k}: " + " / ".join(f"{x:.4f}" for x in v))
