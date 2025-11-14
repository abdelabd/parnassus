# Testing Delphes Efficiency PyTorch Module with ROOT Files

## Quick Start

### Prerequisites

Install required packages:
```bash
pip install uproot awkward torch numpy matplotlib
```

Or with conda:
```bash
conda install -c conda-forge uproot awkward
conda install pytorch numpy matplotlib
```

### Running the Test

1. **Generate a Delphes ROOT file** (if you don't have one):
```bash
cd /pscratch/sd/a/aelabd/sim_software/Delphes-3.5.0
./DelphesHepMC3 cards/delphes_card_CMS_minimal.tcl output.root input.hepmc
```

2. **Run the test script**:
```bash
cd /pscratch/sd/a/aelabd
python test_efficiency_on_root.py output.root
```

Or with limited events:
```bash
python test_efficiency_on_root.py output.root 100  # Process only 100 events
```

## What the Script Does

### 1. **Reads Delphes ROOT File**
   - Opens `output.root` with `uproot`
   - Reads particle data from branches like:
     - `ParticleBeforeProp` (if using minimal card)
     - `ParticleAfterProp` (if using minimal card)
     - `Particle` (if using full card)
   - Extracts: PID, Status, Charge, E, Px, Py, Pz, PT, Eta, Phi, X, Y, Z, T

### 2. **Converts to PyTorch Format**
   - Transforms Delphes data to `(N, 10)` tensor format:
     ```
     [T, X, Y, Z, E, Px, Py, Pz, PDG_ID, Status]
     ```

### 3. **Applies Efficiency Filters**
   - Tests three efficiency formulas:
     - Charged Hadron (CMS card settings)
     - Electron (CMS card settings)
     - Muon (CMS card settings)
   - Applies stochastic filtering (random sampling)

### 4. **Generates Validation Plots**

Creates plots in `efficiency_validation/` directory:

#### **`validation_*.png`** - Distribution Comparisons
- pT distribution (before/after efficiency)
- η distribution (before/after efficiency)
- 2D pT vs η heatmap
- Per-event efficiency
- Particle count per event
- Efficiency histogram

#### **`kinematic_eff_*.png`** - Measured Efficiency
- Efficiency vs pT in different η regions
- Efficiency vs η in different pT regions
- Compare measured values with expected formula

### 5. **Prints Statistics**

Example output:
```
======================================================================
Input Data Summary
======================================================================
Number of events: 50
Particle branch: ParticleBeforeProp
Total particles: 15234
Avg particles/event: 304.7

First event breakdown:
  all            :  312 (100.0%)
  charged        :  156 ( 50.0%)
  neutral        :  156 ( 50.0%)
  electron       :    4 (  1.3%)
  muon           :    2 (  0.6%)
  charged_hadron :  150 ( 48.1%)
  photon         :   78 ( 25.0%)

======================================================================
Testing Efficiency Modules
======================================================================

Charged Hadron Efficiency:
--------------------------------------------------
  Event 0:  312 →  142 ( 45.5%)
  Event 1:  298 →  135 ( 45.3%)
  Event 2:  315 →  148 ( 47.0%)

  Overall: 15234 → 6891 (45.23%)
  Mean event efficiency: 45.18% ± 3.24%
```

## Output Files

After running, you'll find:

```
efficiency_validation/
├── validation_Charged_Hadron.png
├── validation_Electron.png
├── validation_Muon.png
├── kinematic_eff_Charged_Hadron.png
├── kinematic_eff_Electron.png
└── kinematic_eff_Muon.png
```

## Interpretation

### Expected Results

For **Charged Hadrons** with CMS efficiency:
- Central barrel (|η| < 1.5):
  - Low pT (0.1-1 GeV): ~70% efficiency
  - High pT (>1 GeV): ~95% efficiency
- Forward endcap (1.5 < |η| < 2.5):
  - Low pT: ~60% efficiency
  - High pT: ~85% efficiency
- Very forward (|η| > 2.5): 0% efficiency

### Validation

The plots show:
1. **Distribution shifts**: High-pT and central particles preferentially kept
2. **Measured efficiency**: Should match formula within statistical errors
3. **Per-event variation**: Natural stochastic fluctuations

## Example with Real Data

If you have a Delphes output from a physics process (e.g., Z→μμ):

```bash
# Generate events
cd /pscratch/sd/a/aelabd/sim_software/madgraph/
./bin/mg5_aMC  # Generate Z→μμ events

# Run Delphes
cd /pscratch/sd/a/aelabd/sim_software/Delphes-3.5.0
./DelphesHepMC3 cards/delphes_card_CMS.tcl zmumu_output.root Events/run_01/events.hepmc

# Test efficiency module
cd /pscratch/sd/a/aelabd
python test_efficiency_on_root.py zmumu_output.root

# Look for muon efficiency results
# Should see ~99% efficiency for high-pT muons in barrel
```

## Troubleshooting

### Error: "No particle branch found"

Your ROOT file doesn't have expected branches. Check with:
```python
import uproot
file = uproot.open("output.root")
print(file["Delphes"].keys())
```

Solution: Use correct Delphes card that saves particles to TreeWriter.

### Error: "uproot not installed"

Install with:
```bash
pip install uproot awkward
```

### Low efficiency values

Check if you're using the right particle branch:
- `ParticleBeforeProp` = before propagation (use for testing)
- `ParticleAfterProp` = after propagation (already position-dependent)
- `Particle` = all generator particles (includes unstable)

### Plots look wrong

Make sure you're testing on stable particles (Status=1) with reasonable kinematics.

## Advanced Usage

### Test specific particle types

Modify the script to filter by PDG ID:
```python
# In analyze_event function, add filtering:
charged_hadron_mask = ~np.isin(np.abs(event_data['pid']), [11, 13])
particles = particles[charged_hadron_mask]
```

### Compare with Delphes output

If your card includes efficiency modules, compare:
```bash
# Run Delphes WITH efficiency modules
./DelphesHepMC3 cards/delphes_card_CMS.tcl with_eff.root input.hepmc

# Run Delphes WITHOUT efficiency (minimal card)
./DelphesHepMC3 cards/delphes_card_CMS_minimal.tcl no_eff.root input.hepmc

# Test PyTorch module on no_eff.root
python test_efficiency_on_root.py no_eff.root

# Compare particle counts in with_eff.root vs PyTorch output
```

### Export filtered events

Modify the script to save filtered particles back to ROOT or HDF5:
```python
# After applying efficiency
import h5py
with h5py.File('filtered_particles.h5', 'w') as f:
    f.create_dataset('particles', data=filtered.cpu().numpy())
```

## Performance

Typical performance:
- **10 events × 300 particles**: ~50 ms
- **100 events × 300 particles**: ~300 ms
- **1000 events × 300 particles**: ~2 s

Use GPU for large datasets:
```python
eff_module = DelphesEfficiency(device='cuda')
```

## Citation

If you use this code, please cite both Delphes and this module (see README_efficiency.md).
