#!/usr/bin/env python3
"""
Quick verification script to check if all dependencies are installed
and the efficiency module works correctly.

Run this first before testing on ROOT files.
"""

import sys

print("="*70)
print("Verifying Delphes Efficiency PyTorch Module Installation")
print("="*70)
print()

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# Check dependencies
dependencies = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'uproot': 'uproot (for ROOT files)',
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {name} installed")
    except ImportError:
        print(f"✗ {name} NOT installed")
        missing.append(module)

if missing:
    print()
    print("Missing dependencies! Install with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

print()
print("="*70)
print("Testing delphes_efficiency_pytorch module")
print("="*70)
print()

# Import and test the module
try:
    from parnassus.src.parnassus.torch_delphes.Efficiency import DelphesEfficiency
    print("✓ Module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import module: {e}")
    print("\nMake sure delphes_efficiency_pytorch.py is in the current directory:")
    print("  cd /pscratch/sd/a/aelabd")
    sys.exit(1)

import torch
import numpy as np

# Test 1: Create module
print("\n1. Creating efficiency module...")
try:
    eff_module = DelphesEfficiency(efficiency_formula='charged_hadron_cms')
    print("   ✓ Module created successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Create test particles
print("\n2. Creating test particles...")
n_particles = 100
particles = torch.zeros((n_particles, 10))

# Random kinematics
pt = torch.rand(n_particles) * 50 + 0.1  # 0.1 to 50 GeV
eta = torch.rand(n_particles) * 6 - 3     # -3 to 3
phi = torch.rand(n_particles) * 2 * np.pi - np.pi

# Convert to 4-momentum
particles[:, 5] = pt * torch.cos(phi)  # px
particles[:, 6] = pt * torch.sin(phi)  # py
particles[:, 7] = pt * torch.sinh(eta)  # pz
particles[:, 4] = torch.sqrt(particles[:, 5]**2 + particles[:, 6]**2 + 
                             particles[:, 7]**2)  # E
particles[:, 8] = 211  # charged pion
particles[:, 9] = 1    # stable

print(f"   ✓ Created {n_particles} test particles")

# Test 3: Apply efficiency
print("\n3. Applying efficiency filter...")
try:
    filtered, mask = eff_module(particles, return_mask=True)
    n_passed = mask.sum().item()
    efficiency = n_passed / n_particles
    print(f"   ✓ Efficiency applied: {n_passed}/{n_particles} passed ({efficiency*100:.1f}%)")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Check efficiency values
print("\n4. Checking efficiency values...")

# Test specific kinematics
test_cases = [
    (0.05, 0.0, 0.00, "Very low pT"),
    (0.5, 0.5, 0.70, "Low pT, central"),
    (10.0, 0.5, 0.95, "High pT, central"),
    (10.0, 2.0, 0.85, "High pT, forward"),
    (10.0, 3.0, 0.00, "Outside acceptance"),
]

all_correct = True
for pt_test, eta_test, expected_eff, description in test_cases:
    pt_tensor = torch.tensor([pt_test])
    eta_tensor = torch.tensor([eta_test])
    computed_eff = eff_module.efficiency_func(pt_tensor, eta_tensor).item()
    
    if abs(computed_eff - expected_eff) < 0.01:
        status = "✓"
    else:
        status = "✗"
        all_correct = False
    
    print(f"   {status} {description}: expected={expected_eff:.2f}, got={computed_eff:.2f}")

if not all_correct:
    print("\n   ⚠ Some efficiency values don't match expectations!")
    sys.exit(1)

# Test 5: Different formulas
print("\n5. Testing different efficiency formulas...")
formulas = ['charged_hadron_cms', 'electron_cms', 'muon_cms']
for formula in formulas:
    try:
        eff_mod = DelphesEfficiency(efficiency_formula=formula)
        filtered = eff_mod(particles)
        print(f"   ✓ {formula}: {len(filtered)} particles passed")
    except Exception as e:
        print(f"   ✗ {formula} failed: {e}")
        sys.exit(1)

# Test 6: Batched processing
print("\n6. Testing batched processing...")
batch_size = 5
batched_particles = torch.stack([particles] * batch_size)
print(f"   Input shape: {batched_particles.shape}")

try:
    filtered_list = eff_module(batched_particles)
    print(f"   ✓ Processed {len(filtered_list)} batches")
    for i, filtered in enumerate(filtered_list):
        print(f"      Batch {i}: {len(filtered)} particles")
except Exception as e:
    print(f"   ✗ Batched processing failed: {e}")
    sys.exit(1)

# Test 7: Efficiency map generation
print("\n7. Testing efficiency map generation...")
try:
    pt_grid, eta_grid, eff_map = eff_module.get_efficiency_map(
        pt_range=(0, 50),
        eta_range=(-3, 3),
        n_pts=50,
        n_etas=50
    )
    print(f"   ✓ Generated efficiency map: {eff_map.shape}")
    print(f"      Min efficiency: {eff_map.min():.3f}")
    print(f"      Max efficiency: {eff_map.max():.3f}")
    print(f"      Mean efficiency: {eff_map.mean():.3f}")
except Exception as e:
    print(f"   ✗ Efficiency map generation failed: {e}")
    sys.exit(1)

# Test 8: Stochastic behavior
print("\n8. Testing stochastic behavior...")
n_runs = 100
efficiencies = []

for _ in range(n_runs):
    _, mask = eff_module(particles, return_mask=True)
    efficiencies.append(mask.float().mean().item())

mean_eff = np.mean(efficiencies)
std_eff = np.std(efficiencies)

print(f"   ✓ {n_runs} runs completed")
print(f"      Mean efficiency: {mean_eff*100:.2f}%")
print(f"      Std deviation: {std_eff*100:.2f}%")
print(f"      Expected stochastic variation: 3-5%")

if std_eff > 0.01:
    print("   ✓ Stochastic behavior confirmed")
else:
    print("   ⚠ Warning: Low variation (might be deterministic mode)")

print()
print("="*70)
print("All Tests Passed! ✓")
print("="*70)
print()
print("Module is ready to use!")
print()
print("Next steps:")
print("  1. Generate or obtain a Delphes ROOT file")
print("  2. Run: python test_efficiency_on_root.py output.root")
print("  3. Check validation plots in efficiency_validation/")
print()
print("For more information, see:")
print("  - README_efficiency.md")
print("  - TEST_GUIDE.md")
print("  - TESTING_SUMMARY.md")
print()
