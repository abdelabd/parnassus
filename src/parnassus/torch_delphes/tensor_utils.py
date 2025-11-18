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
from typing import Dict, List, Tuple


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
ETA = 8        # Pseudorapidity (position-based EtaOuter for efficiency, momentum-based for smearing)
PHI = 9        # Azimuthal angle
T = 10         # Time
X = 11         # X position
Y = 12         # Y position
Z = 13         # Z position
MASS = 14      # Mass (not stored in Track, will be computed/set based on PID)

# Number of features per particle
N_FEATURES = 15


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


def root_track_to_tensor(track_arrays: ak.Array, branch_prefix: str, 
                        max_events: int = None) -> List[torch.Tensor]:
    """
    Convert ROOT Track branch to list of PyTorch tensors (one per event).
    
    Track objects come from ParticlePropagator and have these attributes:
    - PID, Charge, P, PT, Eta, EtaOuter, Phi, T, X, Y, Z (11 attributes)
    
    We expand this to 15 columns for uniform processing:
    - Columns 0-14: PID, Status, Charge, E, Px, Py, Pz, PT, Eta, Phi, T, X, Y, Z, Mass
    
    Args:
        track_arrays: awkward array with track data from uproot
        branch_prefix: branch name without attribute (e.g., "ChargedHadron")
        max_events: maximum number of events to process
        
    Returns:
        List of tensors, one per event, each of shape (n_particles, 15)
    """
    # Determine number of events
    # Field names in awkward array are "BranchName/BranchName.Attribute"
    n_events = len(track_arrays[f"{branch_prefix}/{branch_prefix}.PT"])
    if max_events is not None:
        n_events = min(n_events, max_events)
    
    event_tensors = []
    
    for i in range(n_events):
        n_particles = len(track_arrays[f"{branch_prefix}/{branch_prefix}.PT"][i])
        
        if n_particles == 0:
            # Empty event - create empty tensor
            event_tensors.append(torch.zeros((0, N_FEATURES), dtype=torch.float64))
            continue
        
        # Create tensor for this event
        particles = np.zeros((n_particles, N_FEATURES))
        
        # Extract raw data from ROOT
        # Field names are "BranchName/BranchName.Attribute"
        pid = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.PID"][i])
        charge = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.Charge"][i])
        p = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.P"][i])
        pt = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.PT"][i])
        eta = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.Eta"][i])        # Momentum eta
        eta_outer = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.EtaOuter"][i])  # Position eta
        phi = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.Phi"][i])
        t = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.T"][i])
        x = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.X"][i])
        y = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.Y"][i])
        z = np.array(track_arrays[f"{branch_prefix}/{branch_prefix}.Z"][i])
        
        # Fill tensor columns
        particles[:, PID] = pid
        particles[:, STATUS] = 1  # Tracks are always status 1
        particles[:, CHARGE] = charge
        particles[:, E] = p  # Approximate E ≈ P for high-energy particles
        particles[:, PX] = pt * np.cos(phi)
        particles[:, PY] = pt * np.sin(phi)
        particles[:, PZ] = pt * np.sinh(eta)  # Use momentum eta for Pz
        particles[:, PT] = pt
        particles[:, ETA] = eta_outer  # Use position eta (EtaOuter) for efficiency calculation
        particles[:, PHI] = phi
        particles[:, T] = t
        particles[:, X] = x
        particles[:, Y] = y
        particles[:, Z] = z
        
        # Estimate mass based on PID (needed for momentum smearing)
        # For most particles we'll use a simple lookup
        masses = np.zeros(n_particles)
        for j, p_id in enumerate(pid):
            abs_pid = abs(int(p_id))
            if abs_pid == ELECTRON_PDG:
                masses[j] = ELECTRON_MASS
            elif abs_pid == MUON_PDG:
                masses[j] = MUON_MASS
            elif abs_pid == CHARGED_PION_PDG:
                masses[j] = PION_MASS
            elif abs_pid == CHARGED_KAON_PDG:
                masses[j] = KAON_MASS
            elif abs_pid == PROTON_PDG:
                masses[j] = PROTON_MASS
            else:
                # Default to pion mass for unknown charged hadrons
                masses[j] = PION_MASS
        
        particles[:, MASS] = masses
        
        # Refine energy calculation: E = sqrt(P^2 + M^2)
        particles[:, E] = np.sqrt(p**2 + masses**2)
        
        # Convert to torch tensor (float64 for precision)
        event_tensors.append(torch.from_numpy(particles))
    
    return event_tensors


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
    # Track attributes we write to ROOT (11 attributes, not all 15 columns)
    attributes = ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z']
    
    # Column indices for each attribute in the tensor
    column_map = {
        'PID': PID,
        'Charge': CHARGE,
        'P': None,  # Will compute from Px, Py, Pz
        'PT': PT,
        'Eta': None,  # Will compute from Px, Py, Pz (momentum eta)
        'EtaOuter': ETA,  # Position eta stored in ETA column
        'Phi': PHI,
        'T': T,
        'X': X,
        'Y': Y,
        'Z': Z
    }
    
    # Build dictionary
    root_dict = {}
    
    for attr in attributes:
        # Extract values for all events
        attr_values = []
        
        for event_tensor in event_tensors:
            if event_tensor.shape[0] == 0:
                # Empty event
                attr_values.append(np.array([], dtype=np.float64))
                continue
            
            event_np = event_tensor.cpu().numpy()
            
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
    import uproot
    
    # Combine all branch dictionaries
    combined_dict = {}
    for branch_data in branches_dict.values():
        combined_dict.update(branch_data)
    
    # Write to ROOT file
    with uproot.recreate(output_file) as f:
        f[tree_name] = combined_dict


def load_all_particle_types(tree, max_events: int = None) -> Tuple[List[torch.Tensor], 
                                                                     List[torch.Tensor], 
                                                                     List[torch.Tensor]]:
    """
    Load all three particle types (ChargedHadron, Electron, Muon) from ROOT tree.
    
    Args:
        tree: uproot TTree object
        max_events: Maximum number of events to load
        
    Returns:
        Tuple of (charged_hadron_tensors, electron_tensors, muon_tensors)
        Each is a list of tensors, one per event
    """
    import uproot
    
    # Define branches and their attribute keys
    branches = {
        'ChargedHadron': ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z'],
        'Electron': ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z'],
        'Muon': ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z']
    }
    
    particle_tensors = {}
    
    for branch_name, attrs in branches.items():
        # Build list of keys for this branch
        branch_keys = [f"{branch_name}/{branch_name}.{attr}" for attr in attrs]
        
        # Read from ROOT
        arrays = tree.arrays(branch_keys, entry_stop=max_events, library="ak")
        
        # Convert to tensors
        tensors = root_track_to_tensor(arrays, branch_name, max_events)
        particle_tensors[branch_name] = tensors
    
    return (particle_tensors['ChargedHadron'], 
            particle_tensors['Electron'], 
            particle_tensors['Muon'])
