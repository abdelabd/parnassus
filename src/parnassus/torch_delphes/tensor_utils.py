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
ETA = 8        # Pseudorapidity (position-based EtaOuter for efficiency, momentum-based for smearing)
PHI = 9        # Azimuthal angle
T = 10         # Time
X = 11         # X position
Y = 12         # Y position
Z = 13         # Z position
MASS = 14      # Mass (not stored in Track, will be computed/set based on PID)

IS_NOT_PAD = 15  # Column index for initial validity mask (1=real, 0=padded)
PASS_PROP = 16
PASS_EFF = 17
PASS_MERGER = 18  # Column index for merger pass mask
PASS_ECAL_TOWER = 19  # Column index for ECal tower mask
PASS_EFLOW_TRACK = 20  # Column index for energy flow track mask
PASS_EFLOW_PHOTON = 21  # Column index for energy flow photon mask

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
    "IS_NOT_PAD": IS_NOT_PAD,
    "PASS_PROP": PASS_PROP,
    "PASS_EFF": PASS_EFF,
    "PASS_MERGER": PASS_MERGER,
    "PASS_ECAL_TOWER": PASS_ECAL_TOWER,
    "PASS_EFLOW_TRACK": PASS_EFLOW_TRACK,
    "PASS_EFLOW_PHOTON": PASS_EFLOW_PHOTON
}

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


# ==================== BATCHING UTILITIES ====================

def compute_max_particles(event_tensors: List[torch.Tensor], scale: float = 1.2) -> int:
    """
    Compute max_particles for padding as scale * max particle count in dataset.
    
    Args:
        event_tensors: List of (N_i, 15) tensors
        scale: Scaling factor (default 1.2 = 20% buffer)
        
    Returns:
        max_particles: Integer max particles for padding
    """
    if len(event_tensors) == 0:
        return 0
    max_count = max(t.shape[0] for t in event_tensors)
    return int(max_count * scale)


def pad_and_batch(event_tensors: List[torch.Tensor], max_particles: int) -> torch.Tensor:
    """
    Pad events to max_particles and stack into batch with mask.
    
    The mask is appended as column 15 to indicate real vs padded particles.
    
    Args:
        event_tensors: List of (N_i, 15) tensors
        max_particles: Max particles to pad to
        
    Returns:
        batch: (B, max_particles, 16) where:
               - batch[:, :, :15] = particle data (padded with zeros)
               - batch[:, :, 15] = mask (1.0 for real particles, 0.0 for padding)
    """
    batch_size = len(event_tensors)
    
    # Get dtype and device from first tensor
    dtype = event_tensors[0].dtype
    device = event_tensors[0].device
    
    # Create padded batch tensor (B, max_particles, 16)
    # Initialize with zeros (padding)
    batch = torch.zeros((batch_size, max_particles, 16), dtype=dtype, device=device)
    
    for i, event in enumerate(event_tensors):
        n_particles = event.shape[0]
        if n_particles > max_particles:
            # Truncate if exceeds max_particles
            batch[i, :max_particles, :15] = event[:max_particles]
            batch[i, :max_particles, 15] = 1.0
        else:
            # Pad with zeros
            batch[i, :n_particles, :15] = event
            batch[i, :n_particles, 15] = 1.0  # Mask for real particles
            # Rest is already zeros (padding)
    
    return batch

def zero_pad_to_max_particles(event_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    max_particles must be greater than or equal to the largest event in event_tensors.

    Pad events to max_particles and stack into batch with mask.
    
    The mask is appended as column 15 to indicate real vs padded particles.
    
    Args:
        event_tensors: List of (N_i, 15) tensors
        max_particles: Max particles to pad to
        
    Returns:
        batch: (B, max_particles, 16) where:
               - batch[:, :, :15] = particle data (padded with zeros)
               - batch[:, :, 15] = mask (1.0 for real particles, 0.0 for padding)
    """

    n_events = len(event_tensors)
    max_particles = compute_max_particles(event_tensors)
    dtype = event_tensors[0].dtype
    device = event_tensors[0].device
    
    # Create padded batch tensor (B, max_particles, 16)
    # Initialize with zeros (padding)
    padded_events = torch.zeros((n_events, max_particles, 16), dtype=dtype, device=device)
    
    for i, event in enumerate(event_tensors):
        n_particles = event.shape[0]
        padded_events[i, :n_particles, :15] = event
        padded_events[i, :n_particles, 15] = 1.0  # Mask for real particles
        # Rest is already zeros (padding)

    return padded_events


def unbatch_and_unpad(batch: torch.Tensor, mask_col: int = 15) -> List[torch.Tensor]:
    """
    Remove padding using mask column and split batch into list of events.
    
    Args:
        batch: (B, N, 15+K) where column mask_col is the mask
        mask_col: Column index containing the mask (default 15)
        
    Returns:
        List of (N_i, 15) tensors with padding removed and mask column dropped
    """
    batch_size = batch.shape[0]
    event_tensors = []
    
    for i in range(batch_size):
        # Extract mask for this event
        mask = batch[i, :, mask_col] > 0.5  # Boolean mask
        
        # Extract real particles (remove padding)
        real_particles = batch[i, mask, :15]  # Take only data columns, drop mask
        
        event_tensors.append(real_particles)
    
    return event_tensors


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


def root_genparticle_to_tensor(particle_arrays: ak.Array, branch_prefix: str = "Particle",
                                max_events: int = None) -> List[torch.Tensor]:
    """
    Convert ROOT GenParticle branch to list of PyTorch tensors (one per event).
    
    GenParticle objects (from Delphes/stableParticles before PropParticlegator) have:
    - PID, Status, Charge, E, Px, Py, Pz, PT, Eta, Phi, T, X, Y, Z, Mass (15 attributes)
    
    Args:
        particle_arrays: awkward array with particle data from uproot
        branch_prefix: branch name (default: "Particle")
        max_events: maximum number of events to process
        
    Returns:
        List of tensors, one per event, each of shape (n_particles, 15)
    """
    # Determine number of events
    n_events = len(particle_arrays[f"{branch_prefix}/{branch_prefix}.PT"])
    if max_events is not None:
        n_events = min(n_events, max_events)
    
    event_tensors = []
    
    for i in range(n_events):
        n_particles = len(particle_arrays[f"{branch_prefix}/{branch_prefix}.PT"][i])
        
        if n_particles == 0:
            event_tensors.append(torch.zeros((0, N_FEATURES), dtype=torch.float64))
            continue
        
        # Create tensor for this event
        particles = np.zeros((n_particles, N_FEATURES))
        
        # Extract data from ROOT (GenParticle has all 15 attributes)
        particles[:, PID] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.PID"][i])
        particles[:, STATUS] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Status"][i])
        particles[:, CHARGE] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Charge"][i])
        particles[:, E] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.E"][i])
        particles[:, PX] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Px"][i])
        particles[:, PY] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Py"][i])
        particles[:, PZ] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Pz"][i])
        particles[:, PT] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.PT"][i])
        particles[:, ETA] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Eta"][i])
        particles[:, PHI] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Phi"][i])
        particles[:, T] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.T"][i])
        particles[:, X] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.X"][i])
        particles[:, Y] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Y"][i])
        particles[:, Z] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Z"][i])
        particles[:, MASS] = np.array(particle_arrays[f"{branch_prefix}/{branch_prefix}.Mass"][i])
        
        # Convert to torch tensor
        event_tensors.append(torch.from_numpy(particles))
    
    return event_tensors


def load_genparticles(tree, max_events: int = None) -> List[torch.Tensor]:
    """
    Load GenParticle (Delphes/stableParticles) from ROOT tree.
    
    Args:
        tree: uproot TTree object
        max_events: Maximum number of events to load
        
    Returns:
        List of tensors, one per event, each of shape (n_particles, 15)
    """
    branch_name = "Particle"
    attrs = ['PID', 'Status', 'Charge', 'E', 'Px', 'Py', 'Pz', 'PT', 'Eta', 'Phi', 'T', 'X', 'Y', 'Z', 'Mass']
    
    # Build list of keys
    branch_keys = [f"{branch_name}/{branch_name}.{attr}" for attr in attrs]
    
    # Read from ROOT
    arrays = tree.arrays(branch_keys, entry_stop=max_events, library="ak")
    
    # Convert to tensors
    return root_genparticle_to_tensor(arrays, branch_name, max_events)


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
                px = momentum.px
                py = momentum.py
                pz = momentum.pz
                
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
                # PT (col 7) = 0 - will be computed by ParticlePropagator
                # ETA (col 8) = 0 - will be computed by ParticlePropagator  
                # PHI (col 9) = 0 - will be computed by ParticlePropagator
                particles[i, T] = t
                particles[i, X] = x / 10.0  # Convert mm to cm
                particles[i, Y] = y / 10.0
                particles[i, Z] = z / 10.0
                particles[i, MASS] = mass
            
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


