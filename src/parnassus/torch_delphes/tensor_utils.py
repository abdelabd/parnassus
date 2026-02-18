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
EVENT_NUMBER = 17

IS_NOT_PAD = 18  # Column index for initial validity mask (1=real, 0=padded)
PASS_PROP = 19
PASS_ECAL_TOWER = 20  # Column index for ECal tower mask
PASS_EFLOW_TRACK = 21  # Column index for energy flow track mask
PASS_EFLOW_PHOTON = 22  # Column index for energy flow photon mask
TRACK_RESOLUTION = 23  # Track momentum resolution (set by MomentumSmearing module)

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
    "EVENT_NUMBER": EVENT_NUMBER,
    "IS_NOT_PAD": IS_NOT_PAD,
    "PASS_PROP": PASS_PROP,
    "PASS_ECAL_TOWER": PASS_ECAL_TOWER,
    "PASS_EFLOW_TRACK": PASS_EFLOW_TRACK,
    "PASS_EFLOW_PHOTON": PASS_EFLOW_PHOTON,
    "TRACK_RESOLUTION": TRACK_RESOLUTION
}

# Number of features per particle
N_FEATURES = max(COLUMN_MAP.values()) + 1


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
    
    The mask is filled in on column IS_NOT_PAD to indicate real vs padded particles.
    
    Args:
        event_tensors: List of (N_i, N_FEATURES) tensors
        max_particles: Max particles to pad to
        
    Returns:
        batch: (B, max_particles, N_FEATURES) where:
               - batch[:, :, :IS_NOT_PAD] = particle data (padded with zeros)
               - batch[:, :, IS_NOT_PAD] = mask (1.0 for real particles, 0.0 for padding)
    """

    n_events = len(event_tensors)
    max_particles = compute_max_particles(event_tensors)
    dtype = torch.float64 
    device = event_tensors[0].device
    
    # Create padded batch tensor (B, max_particles, N_FEATURES+1)
    # Initialize with zeros (padding)
    padded_events = torch.zeros((n_events, max_particles, N_FEATURES), dtype=dtype, device=device)
    
    for i, event in enumerate(event_tensors):
        n_particles = event.shape[0]
        padded_events[i, :n_particles, :] = event
        padded_events[i, :n_particles, IS_NOT_PAD] = 1.0  # Mask for real particles
        # Rest is already zeros (padding)

    return padded_events


def tensor_to_root_dict(batch_tensors: List[torch.Tensor], branch_name: str, 
                        expected_event_numbers: List[float] = None) -> Dict[str, ak.Array]:
    """
    Convert list of event tensors to ROOT-compatible dictionary of awkward arrays.
    
    This creates the structure needed for writing to ROOT files with uproot.
    
    Args:
        batch_tensors: List of tensors, one per batch, each of shape (n_particles, N_FEATURES)
        branch_name: Name for the branch (e.g., "ChargedHadronEfficiency")
        expected_event_numbers: List of all expected event numbers. If provided, ensures
                               all events are represented (with empty arrays for missing events).
        
    Returns:
        Dictionary with keys like "BranchName/BranchName.Attribute" → awkward array
    """
    # Determine if this is a Tower object, Track object, or ParticleFlowCandidate based on branch name
    # Tower objects: Tower, EFlowPhoton (ECal), EFlowNeutralHadron (HCal)
    # ParticleFlowCandidate: EFlowObject (combines Track and Tower fields)
    is_tower = any(keyword in branch_name for keyword in ['Tower', 'EFlowPhoton', 'EFlowNeutralHadron']) and 'EFlowObject' not in branch_name
    is_eflow = 'EFlowObject' in branch_name

    if is_eflow:
        # ParticleFlowCandidate attributes: combination of Track and Tower fields
        # This matches the ParticleFlowCandidate class in DelphesClasses.h (lines 532-613)
        # Track fields: PID, Charge, E, P, PT, Eta, Phi, CtgTheta, C, Mass, EtaOuter, PhiOuter,
        #               T, X, Y, Z, TOuter, XOuter, YOuter, ZOuter, Xd, Yd, Zd, L, D0, DZ,
        #               Nclusters, dNdx, ErrorP, ErrorPT, ErrorPhi, ErrorCtgTheta, ErrorT,
        #               ErrorD0, ErrorDZ, ErrorC, ErrorD0Phi, ErrorD0C, ErrorD0DZ, ErrorD0CtgTheta,
        #               ErrorPhiC, ErrorPhiDZ, ErrorPhiCtgTheta, ErrorCDZ, ErrorCCtgTheta,
        #               ErrorDZCtgTheta, VertexIndex
        # Tower fields: NTimeHits, Eem, Ehad, Edges[4]
        # Note: For Track objects, Tower fields are zero. For Tower objects, Track-specific fields are zero.
        attributes = ['PID', 'Charge', 'E', 'P', 'PT', 'Eta', 'Phi', 'T', 'X', 'Y', 'Z', 'Eem', 'Ehad']

        # Column indices for ParticleFlowCandidate attributes
        column_map = {
            'PID': PID,
            'Charge': CHARGE,
            'E': E,
            'P': None,  # Will compute from Px, Py, Pz
            'PT': PT,
            'Eta': ETA,
            'Phi': PHI,
            'T': T,
            'X': X,
            'Y': Y,
            'Z': Z,
            'Eem': None,  # Will be zero for Track objects (towers don't have this in tensor)
            'Ehad': None,  # Will be zero for Track objects (towers don't have this in tensor)
        }
    elif is_tower:
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
    
    # First, collect all particles grouped by event number
    # Concatenate all batches into a single tensor
    if len(batch_tensors) == 0 or all(b.shape[0] == 0 for b in batch_tensors):
        # No particles at all - create empty arrays for all expected events
        all_particles = None
        all_event_numbers = set()
    else:
        all_particles = torch.cat([b for b in batch_tensors if b.shape[0] > 0], dim=0)
        all_event_numbers = set(torch.unique(all_particles[:, EVENT_NUMBER]).cpu().numpy().tolist())
    
    # Determine which event numbers to iterate over
    if expected_event_numbers is not None:
        event_nums_to_process = sorted(expected_event_numbers)
    else:
        event_nums_to_process = sorted(all_event_numbers)
    
    # TODO: Fix manual computations
    for attr in attributes:
        # Extract values for all events
        attr_values = []
        
        for event_num in event_nums_to_process:
            if all_particles is None or event_num not in all_event_numbers:
                # Empty event - no particles of this type
                attr_values.append(np.array([], dtype=np.float64))
                continue
            
            event_tensor = all_particles[all_particles[:, EVENT_NUMBER] == event_num]
        
            if event_tensor.shape[0] == 0:
                # Empty event
                attr_values.append(np.array([], dtype=np.float64))
                continue
            
            event_np = event_tensor.cpu().numpy()

            if is_eflow:
                # ParticleFlowCandidate-specific computations
                # Note: EFlowMerger has already applied transformations:
                # - Track Eta → EtaOuter (position eta)
                # - Tower PID set correctly (22 for photons, 0 for neutral hadrons)
                # - Tower X/Y/Z set to zero
                if attr == 'P':
                    # Compute P = sqrt(Px^2 + Py^2 + Pz^2)
                    px = event_np[:, PX]
                    py = event_np[:, PY]
                    pz = event_np[:, PZ]
                    values = np.sqrt(px**2 + py**2 + pz**2)
                elif attr == 'Eta':
                    # EFlowMerger already set Eta to position eta for both tracks and towers
                    # Just use ETA column directly
                    values = event_np[:, ETA]
                elif attr == 'Eem':
                    # Electromagnetic energy: E for photons (PID=22), 0 otherwise
                    pid_vals = event_np[:, PID]
                    e_vals = event_np[:, E]
                    values = np.where(pid_vals == 22, e_vals, 0.0)
                elif attr == 'Ehad':
                    # Hadronic energy: E for neutral hadrons (PID=0), 0 otherwise
                    pid_vals = event_np[:, PID]
                    e_vals = event_np[:, E]
                    values = np.where(pid_vals == 0, e_vals, 0.0)
                elif attr == "T":
                    values = event_np[:, T]*1e-3 / 299792458.0  # Convert mm/c to microseconds
                elif attr == 'PID':
                    # PID should be integer
                    values = event_np[:, column_map[attr]].astype(np.int32)
                else:
                    # Direct extraction
                    values = event_np[:, column_map[attr]]
            elif is_tower:
                # Tower-specific computations
                if attr == 'ET':
                    # ET = E / cosh(Eta)
                    e = event_np[:, E]
                    eta = event_np[:, ETA]
                    values = e / np.cosh(eta)
                elif attr == 'T':
                    # Convert mm/c to seconds (same as tracks)
                    values = event_np[:, T]*1e-3 / 299792458.0
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
                elif attr == "T":
                    values = event_np[:, T]*1e-3 / 299792458.0  # Convert mm/c to microseconds
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
        Batched tensor of shape (n_events, max_particles, N_FEATURES)
    """
    
    event_tensors = []
    
    with pyhepmc.open(hepmc_file) as f:
        for event_idx, event in tqdm(enumerate(f), total=max_events):
            if max_events is not None and event_idx >= max_events:
                break
            
            # Get all stable particles (status == 1)
            event_number = event.event_number
            stable_particles = [p for p in event.particles if p.status == 1]
            
            n_particles = len(stable_particles)
            if n_particles == 0:
                event_tensors.append(torch.zeros((0, N_FEATURES), dtype=torch.float64))
                continue
            
            # ===== VECTORIZED EXTRACTION =====
            # Extract all PIDs at once
            pids = np.array([p.pid for p in stable_particles], dtype=np.int64)
            
            # Extract all momenta at once: (n_particles, 4) for [e, px, py, pz]
            momenta = np.array([[p.momentum.e, p.momentum.px, p.momentum.py, p.momentum.pz] 
                               for p in stable_particles], dtype=np.float64)
            
            # Extract all vertices at once: (n_particles, 4) for [x, y, z, t]
            vertices = np.array([
                [p.production_vertex.position.x, p.production_vertex.position.y,
                 p.production_vertex.position.z, p.production_vertex.position.t]
                if p.production_vertex else [0.0, 0.0, 0.0, 0.0]
                for p in stable_particles
            ], dtype=np.float64)
            
            # ===== VECTORIZED COMPUTATIONS =====
            e = momenta[:, 0]
            px = momenta[:, 1]
            py = momenta[:, 2]
            pz = momenta[:, 3]
            
            pt = np.sqrt(px**2 + py**2)
            eta = np.arcsinh(pz / (pt + 1e-10))
            phi = np.arctan2(py, px)
            
            # Vectorized charge and mass lookup
            charges = get_charges_vectorized(pids)
            masses = get_masses_vectorized(pids)
            
            # ===== BUILD TENSOR IN ONE SHOT =====
            particles = np.zeros((n_particles, N_FEATURES), dtype=np.float64)
            particles[:, PID] = pids
            particles[:, STATUS] = 1  # All stable particles have status 1
            particles[:, CHARGE] = charges
            particles[:, E] = e
            particles[:, PX] = px
            particles[:, PY] = py
            particles[:, PZ] = pz
            particles[:, PT] = pt
            particles[:, ETA] = eta
            particles[:, PHI] = phi
            particles[:, T] = vertices[:, 3]
            particles[:, X] = vertices[:, 0]
            particles[:, Y] = vertices[:, 1]
            particles[:, Z] = vertices[:, 2]
            particles[:, MASS] = masses
            particles[:, EVENT_NUMBER] = event_number
            # ETA_OUTER, PHI_OUTER will be computed by ParticlePropagator
            
            event_tensors.append(torch.from_numpy(particles))
    
    return zero_pad_to_max_particles(event_tensors)


# ==================== PDG ID UTILITIES (VECTORIZED) ====================

def get_charges_vectorized(pids: np.ndarray) -> np.ndarray:
    """
    Get electric charges for an array of PDG IDs (vectorized).
    
    Args:
        pids: NumPy array of PDG IDs
        
    Returns:
        NumPy array of electric charges
    """
    abs_pids = np.abs(pids)
    signs = np.sign(pids)
    charges = np.zeros_like(pids, dtype=np.float64)
    
    # Leptons (electron, muon, tau): charge = -sign(pid)
    lepton_mask = (abs_pids == 11) | (abs_pids == 13) | (abs_pids == 15)
    charges[lepton_mask] = -signs[lepton_mask]
    
    # Neutrinos: charge = 0 (already initialized)
    
    # Photon: charge = 0 (already initialized)
    
    # Charged pions: charge = sign(pid)
    charges[abs_pids == 211] = signs[abs_pids == 211]
    
    # Neutral pion: charge = 0 (already initialized)
    
    # Charged kaons: charge = sign(pid)
    charges[abs_pids == 321] = signs[abs_pids == 321]
    
    # Neutral kaons: charge = 0 (already initialized)
    
    # Proton: charge = sign(pid)
    charges[abs_pids == 2212] = signs[abs_pids == 2212]
    
    # Neutron: charge = 0 (already initialized)
    
    # Lambda baryon (3122): charge = 0 (already initialized)
    
    # Sigma baryons
    charges[abs_pids == 3222] = signs[abs_pids == 3222]   # Sigma+
    charges[abs_pids == 3112] = -signs[abs_pids == 3112]  # Sigma-
    # Sigma0 (3212): charge = 0 (already initialized)
    
    # Xi baryons
    charges[abs_pids == 3312] = -signs[abs_pids == 3312]  # Xi-
    # Xi0 (3322): charge = 0 (already initialized)
    
    # Omega baryon
    charges[abs_pids == 3334] = -signs[abs_pids == 3334]  # Omega-
    
    # D mesons
    charges[abs_pids == 411] = signs[abs_pids == 411]   # D+
    charges[abs_pids == 421] = 0  # D0
    charges[abs_pids == 431] = signs[abs_pids == 431]   # Ds+
    
    # B mesons
    charges[abs_pids == 521] = signs[abs_pids == 521]   # B+
    charges[abs_pids == 511] = 0  # B0
    charges[abs_pids == 531] = 0  # Bs0
    
    return charges


def get_masses_vectorized(pids: np.ndarray) -> np.ndarray:
    """
    Get particle masses for an array of PDG IDs (vectorized).
    
    Args:
        pids: NumPy array of PDG IDs
        
    Returns:
        NumPy array of masses in GeV
    """
    abs_pids = np.abs(pids)
    
    # Default to pion mass for unknown particles
    masses = np.full_like(pids, PION_MASS, dtype=np.float64)
    
    # Leptons
    masses[abs_pids == 11] = ELECTRON_MASS
    masses[abs_pids == 13] = MUON_MASS
    masses[abs_pids == 15] = 1.77686  # tau
    
    # Neutrinos
    masses[(abs_pids == 12) | (abs_pids == 14) | (abs_pids == 16)] = 0.0
    
    # Photon
    masses[abs_pids == 22] = 0.0
    
    # Pions
    masses[abs_pids == 111] = 0.13498  # pi0
    masses[abs_pids == 211] = PION_MASS  # pi+/-
    
    # Kaons
    masses[(abs_pids == 130) | (abs_pids == 310) | (abs_pids == 311)] = 0.49761  # neutral kaons
    masses[abs_pids == 321] = KAON_MASS  # K+/-
    
    # Nucleons
    masses[abs_pids == 2212] = PROTON_MASS
    masses[abs_pids == 2112] = 0.93957  # neutron
    
    # Lambda
    masses[abs_pids == 3122] = 1.11568
    
    # Sigma baryons
    masses[abs_pids == 3222] = 1.18937  # Sigma+
    masses[abs_pids == 3212] = 1.19264  # Sigma0
    masses[abs_pids == 3112] = 1.19745  # Sigma-
    
    # Xi baryons
    masses[abs_pids == 3322] = 1.31486  # Xi0
    masses[abs_pids == 3312] = 1.32171  # Xi-
    
    # Omega
    masses[abs_pids == 3334] = 1.67245
    
    # D mesons
    masses[abs_pids == 411] = 1.86966  # D+
    masses[abs_pids == 421] = 1.86484  # D0
    masses[abs_pids == 431] = 1.96835  # Ds+
    
    # B mesons
    masses[abs_pids == 521] = 5.27934  # B+
    masses[abs_pids == 511] = 5.27965  # B0
    masses[abs_pids == 531] = 5.36688  # Bs0
    
    return masses

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
