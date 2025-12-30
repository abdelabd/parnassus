"""
Utility functions for converting between ROOT files and PyTorch tensors.

This module provides:
- Column index constants for the tensor representation
- ROOT → Tensor conversion (per particle type)
- Tensor → ROOT conversion (for writing output files)
"""
import torch
import numpy as np
import awkward as ak
import uproot
import pyhepmc
from typing import Dict, List, Tuple
from tqdm import tqdm


# ==================== TENSOR COLUMN INDICES ====================
# Track objects from ParticlePropagator have 11 attributes.
# We expand this to 15 columns for compatibility with physics calculations.
#
# Columns 0-14 (15 total):
PID = 0        # Particle ID (PDG code)
STATUS = 1     # Status (dummy value for Track objects, always 1)
CHARGE = 2     # Electric charge
E = 3          # Energy (approximated from P for Track objects)
PX = 4         # X-component of momentum
PY = 5         # Y-component of momentum
PZ = 6         # Z-component of momentum
PT = 7         # Transverse momentum
ETA = 8        # Pseudorapidity (momentum)
PHI = 9        # Azimuthal angle (momentum)
T = 10         # Time
X = 11         # X position
Y = 12         # Y position
Z = 13         # Z position
MASS = 14      # Mass (not stored in Track, will be computed/set based on PID)
ETA_OUTER = 15 # Pseudorapidity at point of intersection with detector (position)
PHI_OUTER = 16 # Azimuthal angle at closest-approach to z-axis (position)

IS_NOT_PAD = 17  # Column index for initial validity mask (1=real, 0=padded)
PASS_PROP = 18
PASS_EFF = 19
PASS_MERGER = 20  # Column index for merger pass mask
PASS_ECAL_TOWER = 21  # Column index for ECal tower mask
PASS_EFLOW_TRACK = 22  # Column index for energy flow track mask
PASS_EFLOW_PHOTON = 23  # Column index for energy flow photon mask

COLUMN_MAP = {
    "PID": PID,
    "STATUS": STATUS,
    "CHARGE": CHARGE,
    "E": E,
    "PX": PX,
    "PY": PY,
    "PZ": PZ,
    "PT": PT,
    "ETA": ETA,
    "PHI": PHI,
    "T": T,
    "X": X,
    "Y": Y,
    "Z": Z,
    "MASS": MASS,
    "ETA_OUTER": ETA_OUTER,
    "PHI_OUTER": PHI_OUTER,
    "IS_NOT_PAD": IS_NOT_PAD,
    "PASS_PROP": PASS_PROP,
    "PASS_EFF": PASS_EFF,
    "PASS_MERGER": PASS_MERGER,
    "PASS_ECAL_TOWER": PASS_ECAL_TOWER,
    "PASS_EFLOW_TRACK": PASS_EFLOW_TRACK,
    "PASS_EFLOW_PHOTON": PASS_EFLOW_PHOTON
}

# Number of features per particle
N_FEATURES = 17


# ==================== PDG ID CONSTANTS ====================
# Note: ParticlePropagator separates particles by type, so we may not need
# explicit PDG filtering. But useful for reference and future extensions.
ELECTRON_PDG = 11
MUON_PDG = 13
CHARGED_PION_PDG = 211
CHARGED_KAON_PDG = 321
PROTON_PDG = 2212

# Masses (GeV)
PION_MASS = 0.13957
KAON_MASS = 0.49368
PROTON_MASS = 0.93827
ELECTRON_MASS = 0.000511
MUON_MASS = 0.10566


# ==================== BATCHING UTILITIES ====================

def compute_max_particles(event_tensors: List[torch.Tensor], scale: float = 1.2) -> int:
    """
    Compute max_particles for padding as scale * max particle count in dataset.
    
    Args:
        event_tensors: List of (N_i, N_FEATURES) tensors
        scale: Scaling factor (default 1.2 = 20% buffer)
        
    Returns:
        max_particles: Integer max particles for padding
    """
    if len(event_tensors) == 0:
        return 0
    max_count = max(t.shape[0] for t in event_tensors)
    return int(max_count * scale)


def zero_pad_to_max_particles(event_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    max_particles must be greater than or equal to the largest event in event_tensors.

    Pad events to max_particles and stack into batch with mask.
    
    The mask is appended as column N_FEATURES to indicate real vs padded particles.
    
    Args:
        event_tensors: List of (N_i, N_FEATURES) tensors
        max_particles: Max particles to pad to
        
    Returns:
        batch: (B, max_particles, N_FEATURES+1) where:
               - batch[:, :, :N_FEATURES] = particle data (padded with zeros)
               - batch[:, :, N_FEATURES] = mask (1.0 for real particles, 0.0 for padding)
    """

    n_events = len(event_tensors)
    max_particles = compute_max_particles(event_tensors)
    dtype = event_tensors[0].dtype
    device = event_tensors[0].device
    
    # Create padded batch tensor (B, max_particles, N_FEATURES+1)
    # Initialize with zeros (padding)
    padded_events = torch.zeros((n_events, max_particles, N_FEATURES+1), dtype=dtype, device=device)
    
    for i, event in enumerate(event_tensors):
        n_particles = event.shape[0]
        padded_events[i, :n_particles, :N_FEATURES] = event
        padded_events[i, :n_particles, N_FEATURES] = 1.0  # Mask for real particles
        # Rest is already zeros (padding)

    return padded_events


def tensor_to_root_dict(event_tensors: List[torch.Tensor], branch_name: str) -> Dict[str, ak.Array]:
    """
    Convert list of event tensors to ROOT-compatible dictionary of awkward arrays.
    
    This creates the structure needed for writing to ROOT files with uproot.
    
    Args:
        event_tensors: List of tensors, one per event, each of shape (n_particles, 15)
        branch_name: Name for the branch (e.g., "ChargedHadronEfficiency")
        
    Returns:
        Dictionary with keys like "BranchName/BranchName.Attribute" → awkward array
    """
    # Determine if this is a Tower object or Track object based on branch name
    is_tower = any(keyword in branch_name for keyword in ['Tower', 'EFlowPhoton'])
    
    if is_tower:
        # Tower attributes: E, ET, Eta, Phi, T, Eem, Ehad
        attributes = ['E', 'ET', 'Eta', 'Phi', 'T']
        
        # Column indices for tower attributes
        column_map = {
            'E': E,
            'ET': None,  # Will compute as E / cosh(Eta)
            'Eta': ETA,  # Momentum eta
            'Phi': PHI,  # Momentum phi
            'T': T
        }
    else:
        # Track attributes (existing code)
        attributes = ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z']
        
        # Column indices for each attribute in the tensor
        column_map = {
            'PID': PID,
            'Charge': CHARGE,
            'P': None,  # Will compute from Px, Py, Pz
            'PT': PT,
            'Eta': ETA,  # Will compute from Px, Py, Pz (momentum eta)
            'EtaOuter': ETA_OUTER,  # Position eta stored in ETA_OUTER column
            'Phi': PHI,
            'PhiOuter': PHI_OUTER,  # Position phi stored in PHI_OUTER column
            'T': T,
            'X': X,
            'Y': Y,
            'Z': Z
        }
    
    # Build dictionary
    root_dict = {}
    
    # TODO: Fix manual computations
    for attr in attributes:
        # Extract values for all events
        attr_values = []
        
        for event_tensor in event_tensors:
            if event_tensor.shape[0] == 0:
                # Empty event
                attr_values.append(np.array([], dtype=np.float64))
                continue
            
            event_np = event_tensor.cpu().numpy()
            
            if is_tower:
                # Tower-specific computations
                if attr == 'ET':
                    # ET = E / cosh(Eta)
                    e = event_np[:, E]
                    eta = event_np[:, ETA]
                    values = e / np.cosh(eta)
                elif attr in column_map and column_map[attr] is not None:
                    values = event_np[:, column_map[attr]]
                else:
                    values = np.zeros(event_np.shape[0])
            else:
                # Track-specific computations
                if attr == 'P':
                    # Compute P = sqrt(Px^2 + Py^2 + Pz^2)
                    px = event_np[:, PX]
                    py = event_np[:, PY]
                    pz = event_np[:, PZ]
                    values = np.sqrt(px**2 + py**2 + pz**2) 
                elif attr == 'Eta':
                    # Compute momentum eta from Px, Py, Pz
                    px = event_np[:, PX]
                    py = event_np[:, PY]
                    pz = event_np[:, PZ]
                    pt = np.sqrt(px**2 + py**2) 
                    values = np.arcsinh(pz / (pt + 1e-10))
                elif attr == 'PID':
                    # PID should be integer
                    values = event_np[:, column_map[attr]].astype(np.int32)
                else:
                    # Direct extraction
                    values = event_np[:, column_map[attr]]
            
            attr_values.append(values)
        
        # Convert to awkward array
        ak_array = ak.Array(attr_values)
        
        # Add to dictionary with ROOT branch naming
        key = f"{branch_name}/{branch_name}.{attr}"
        root_dict[key] = ak_array
    
    return root_dict


def write_root_file(output_file: str, branches_dict: Dict[str, Dict[str, ak.Array]], 
                   tree_name: str = "Delphes"):
    """
    Write multiple branches to a ROOT file.
    
    Args:
        output_file: Path to output ROOT file
        branches_dict: Dictionary mapping branch names to their data dictionaries
                      e.g., {"ChargedHadronEfficiency": {...}, "ElectronEfficiency": {...}}
        tree_name: Name of the tree in ROOT file (default: "Delphes")
    """
    
    # Combine all branch dictionaries
    combined_dict = {}
    for branch_data in branches_dict.values():
        combined_dict.update(branch_data)
    
    # Write to ROOT file
    with uproot.recreate(output_file) as f:
        f[tree_name] = combined_dict


def hepmc_to_tensor(hepmc_file: str, max_events: int = None) -> List[torch.Tensor]:

    """
    Convert HepMC file to list of PyTorch tensors (one per event).
    
    Reads stable particles from HepMC events and converts to tensor format.
    
    Args:
        hepmc_file: Path to HepMC file (.hepmc, .hepmc3, or .hepmc.gz)
        max_events: Maximum number of events to process
        
    Returns:
        List of tensors, one per event, each of shape (n_particles, 15)
    """
    import pyhepmc
    
    event_tensors = []
    
    with pyhepmc.open(hepmc_file) as f:
        for event_idx, event in tqdm(enumerate(f), total=max_events):
            if max_events is not None and event_idx >= max_events:
                break
            
            # Get all stable particles (status == 1)
            stable_particles = [p for p in event.particles if p.status == 1]
            
            n_particles = len(stable_particles)
            if n_particles == 0:
                event_tensors.append(torch.zeros((0, N_FEATURES), dtype=torch.float64))
                continue
            
            # Create tensor for this event
            particles = np.zeros((n_particles, N_FEATURES))
            
            for i, p in enumerate(stable_particles):
                # Extract raw particle properties - NO COMPUTATIONS
                # Let modules compute PT, Eta, Phi as needed
                pid = p.pid
                status = p.status
                momentum = p.momentum
                
                # Lookup charge and mass from PDG ID
                charge = get_charge_from_pdg(pid)
                mass = get_mass_from_pdg(pid)
                
                # Extract raw momentum components
                e = momentum.e
                px = torch.tensor(momentum.px)
                py = torch.tensor(momentum.py)
                pz = torch.tensor(momentum.pz)
                pt = torch.sqrt(px**2 + py**2)
                eta = torch.asinh(pz / (pt + 1e-10))
                phi = torch.atan2(py, px)
                
                # Extract production vertex
                if p.production_vertex:
                    vertex = p.production_vertex.position
                    x = vertex.x  # mm in HepMC
                    y = vertex.y
                    z = vertex.z
                    t = vertex.t
                else:
                    x = y = z = t = 0.0
                
                # Fill tensor row with RAW data only
                particles[i, PID] = pid
                particles[i, STATUS] = status
                particles[i, CHARGE] = charge
                particles[i, E] = e
                particles[i, PX] = px
                particles[i, PY] = py
                particles[i, PZ] = pz
                particles[i, PT] = pt
                particles[i, ETA] = eta
                particles[i, PHI] = phi
                particles[i, T] = t
                particles[i, X] = x / 10.0  # Convert mm to cm
                particles[i, Y] = y / 10.0
                particles[i, Z] = z / 10.0
                particles[i, MASS] = mass
                # ETA_OUTER - will be computed by ParticlePropagator  
                # PHI_outer - will be computed by ParticlePropagator
            
            # Convert to torch tensor
            event_tensors.append(torch.from_numpy(particles))
    
    return zero_pad_to_max_particles(event_tensors)

# ==================== PDG ID UTILITIES ====================

def get_charge_from_pdg(pdg_id: int) -> float:
    """
    Get electric charge from PDG ID.
    
    Args:
        pdg_id: Particle Data Group ID code
        
    Returns:
        Electric charge in units of elementary charge
    """
    # Leptons
    if abs(pdg_id) == 11:  # electron
        return -1.0 if pdg_id > 0 else 1.0
    elif abs(pdg_id) == 13:  # muon
        return -1.0 if pdg_id > 0 else 1.0
    elif abs(pdg_id) == 15:  # tau
        return -1.0 if pdg_id > 0 else 1.0
    
    # Neutrinos (neutral)
    elif abs(pdg_id) in [12, 14, 16]:
        return 0.0
    
    # Photon
    elif pdg_id == 22:
        return 0.0
    
    # Pions
    elif abs(pdg_id) == 211:  # charged pion
        return 1.0 if pdg_id > 0 else -1.0
    elif pdg_id == 111:  # neutral pion
        return 0.0
    
    # Kaons
    elif abs(pdg_id) == 321:  # charged kaon
        return 1.0 if pdg_id > 0 else -1.0
    elif pdg_id == 130 or pdg_id == 310 or pdg_id == 311:  # neutral kaons
        return 0.0
    
    # Proton/neutron
    elif pdg_id == 2212:  # proton
        return 1.0
    elif pdg_id == -2212:  # antiproton
        return -1.0
    elif pdg_id == 2112 or pdg_id == -2112:  # neutron/antineutron
        return 0.0
    
    # Quarks (should not appear as stable particles, but just in case)
    elif abs(pdg_id) in [1, 3, 5]:  # d, s, b quarks
        return -1.0/3.0 if pdg_id > 0 else 1.0/3.0
    elif abs(pdg_id) in [2, 4, 6]:  # u, c, t quarks
        return 2.0/3.0 if pdg_id > 0 else -2.0/3.0
    
    # For unknown particles, try to infer from PDG numbering scheme
    # This is a simplified approximation
    else:
        # Mesons (100-999): last digit gives charge
        if 100 <= abs(pdg_id) < 1000:
            last_digit = abs(pdg_id) % 10
            if last_digit == 0:  # neutral
                return 0.0
            elif pdg_id > 0:
                return 1.0 if last_digit in [1, 3, 5] else -1.0
            else:
                return -1.0 if last_digit in [1, 3, 5] else 1.0
        
        # Baryons (1000-9999)
        elif 1000 <= abs(pdg_id) < 10000:
            # Extract quark content (simplified)
            return 1.0 if pdg_id > 0 else -1.0
        
        # Default to neutral for unknown particles
        return 0.0


def get_mass_from_pdg(pdg_id: int) -> float:
    """
    Get particle mass from PDG ID.
    
    Args:
        pdg_id: Particle Data Group ID code
        
    Returns:
        Mass in GeV
    """
    mass_table = {
        # Leptons
        11: ELECTRON_MASS,
        13: MUON_MASS,
        15: 1.77686,  # tau
        12: 0.0,  # electron neutrino
        14: 0.0,  # muon neutrino
        16: 0.0,  # tau neutrino
        
        # Photon
        22: 0.0,
        
        # Pions
        111: 0.13498,  # pi0
        211: PION_MASS,  # pi+/-
        
        # Kaons
        130: 0.49761,  # KL
        310: 0.49761,  # KS
        311: 0.49761,  # K0
        321: KAON_MASS,  # K+/-
        
        # Proton/neutron
        2212: PROTON_MASS,
        2112: 0.93957,  # neutron
    }
    
    abs_pdg = abs(pdg_id)
    if abs_pdg in mass_table:
        return mass_table[abs_pdg]
    
    # Default mass for unknown particles (use pion mass as reasonable default)
    return PION_MASS

