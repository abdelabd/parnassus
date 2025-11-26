"""
PyTorch implementation of Delphes ParticlePropagator module.

Propagates charged and neutral particles from a given vertex to a cylinder
defined by its radius and half-length, centered at (0,0,0) with axis along z.

This module:
1. Propagates neutral particles in straight lines
2. Propagates charged particles in helical paths through magnetic field
3. Separates output into: stableParticles, chargedHadrons, electrons, muons, neutralParticles
"""

import torch
import torch.nn as nn
import numpy as np


class ParticlePropagator(nn.Module):
    """
    PyTorch implementation of Delphes ParticlePropagator module.
    
    Propagates particles from production vertex to detector surface (cylinder).
    Uses masks to separately handle:
    - Neutral particles (straight line propagation)
    - Charged particles (helix propagation in magnetic field)
    
    IMPORTANT: This module computes PT and position-based Eta from raw data:
    - PT (column 7): sqrt(Px² + Py²) 
    - Eta (column 8): asinh(Z / sqrt(X² + Y²)) - POSITION-based, from production vertex
    - Phi (column 9): atan2(Py, Px)
    
    Position-based Eta is used by the Efficiency module (matching C++ Delphes behavior).
    
    Input shape: (N, 15) - GenParticle format where:
        - column 0: PID (Particle ID)
        - column 1: Status
        - column 2: Charge
        - column 3: E (Energy)
        - columns 4-6: Px, Py, Pz (3-momentum)
        - column 7: PT (computed here, initially 0)
        - column 8: Eta (computed here as position-based, initially 0)
        - column 9: Phi (computed here, initially 0)
        - column 10: T (time)
        - columns 11-13: X, Y, Z (position at production vertex)
        - column 14: Mass
    
    Output: Propagated particles in Track format (same 15 columns but with updated positions)
    """
    
    def __init__(self,
                 radius=1.29,           # Detector radius in meters
                 half_length=3.0,       # Detector half-length in meters
                 bz=3.8,                # Magnetic field in Tesla
                 radius_max=None,       # Max radius for initial position check
                 half_length_max=None,  # Max half-length for initial position check
                 device='cpu'):
        """
        Args:
            radius: Detector cylinder radius in meters (default: 1.29m for CMS)
            half_length: Detector cylinder half-length in meters (default: 3.0m)
            bz: Magnetic field strength in Tesla (default: 3.8T for CMS)
            radius_max: Maximum radius for particle origin (default: same as radius)
            half_length_max: Maximum half-length for particle origin (default: same as half_length)
            device: torch device ('cpu' or 'cuda')
        """
        super().__init__()
        self.radius = radius
        self.radius2 = radius * radius
        self.half_length = half_length
        self.bz = torch.tensor(bz, dtype=torch.float64)
        self.radius_max = radius_max if radius_max is not None else radius
        self.half_length_max = half_length_max if half_length_max is not None else half_length
        self.device = device
        
        # Physical constant
        self.c_light = 2.99792458e8  # Speed of light in m/s
        
    def forward(self, particles):
        """
        Propagate particles to detector surface using mask-based filtering.
        
        Args:
            particles: tensor of shape (N, 15) or (B, N, 16)
                If (N, 15): single event, will add mask column
                If (B, N, 16): batched events with mask in column 15
        
        Returns:
            For single event: dict with keys 'ParticleAfterProp', 'ChargedHadron', etc.
                Each value is a tensor of shape (N, 16) where column 15 is the mask
                (1.0 = particle survived, 0.0 = filtered out)
            
            For batched: dict with keys 'ParticleAfterProp', etc., each containing (B, N, 16)
                with updated masks indicating which particles survived propagation
        """
        
        # Detect batched input
        is_batched = particles.ndim == 3
        
        if is_batched:
            # Batched processing - flatten, process, reshape
            batch_size = particles.shape[0]
            max_particles = particles.shape[1]
            
            # Flatten: (B, N, 15/16) -> (B*N, 15/16)
            particles_flat = particles.reshape(-1, particles.shape[-1])
            
            # Process all particles at once
            result_dict_flat = self._process_particles(particles_flat)
            
            # Reshape back to batches: (B*N, 16) -> (B, N, 16)
            output_dict = {}
            for key in result_dict_flat.keys():
                output_dict[key] = result_dict_flat[key].reshape(batch_size, max_particles, 16)
            
            return output_dict
        else:
            # Single event or unbatched particles
            return self._process_particles(particles)
    
    def _process_particles(self, particles):
        """
        Core propagation logic - works on (N, 15) or (N, 16) particles.
        
        This method contains all the actual physics logic and works identically
        whether processing a single event or a flattened batch.
        
        Args:
            particles: (N, 15) or (N, 16) tensor
        
        Returns:
            Dict with keys 'ParticleAfterProp', 'ChargedHadron', etc.
            Each value is a tensor of shape (N, 16) with mask in column 15
        """

        # Extract initial mask (will be updated as we filter)
        mask = particles[:, 15] > 0.5
        
        # Compute PT and Phi from raw momentum components (for all particles)
        px = particles[:, 4]  # Momentum components (GeV)
        py = particles[:, 5]
        pz = particles[:, 6]
        
        # Compute PT from momentum components
        pt = torch.sqrt(px**2 + py**2)
        particles[:, 7] = pt  # Store in column 7
        
        # Compute Phi from momentum
        phi = torch.atan2(py, px)
        particles[:, 9] = phi  # Store in column 9
        
        # NOTE: Eta (column 8) will be computed as POSITION-BASED eta after propagation
        # in _propagate_neutral and _propagate_charged methods
        
        # Extract positions (stored in cm, convert to m for calculations)
        x_cm = particles[:, 11]  # X position in cm
        y_cm = particles[:, 12]  # Y position in cm
        z_cm = particles[:, 13]  # Z position in cm
        x = x_cm * 1.0e-2  # Convert cm to m
        y = y_cm * 1.0e-2
        z = z_cm * 1.0e-2
        t = particles[:, 10]  # Time
        e = particles[:, 3]
        q = particles[:, 2]   # Charge
        
        # Check if particles are within detector volume
        r = torch.sqrt(x**2 + y**2)
        inside_volume = (r <= self.radius_max) & (torch.abs(z) <= self.half_length_max)
        
        # Check minimum PT
        valid_pt = pt**2 >= 1.0e-9
        
        # Update mask: filter out particles that fail these checks
        mask = mask & inside_volume & valid_pt
        
        # ==================== HANDLE "ALREADY OUTSIDE TRACKER" CASE ====================
        # C++ Delphes: if(r > fRadius || |z| > fHalfLength) → pass through without propagation
        inside_tracker = (r <= self.radius) & (torch.abs(z) <= self.half_length)
        needs_propagation = inside_tracker & mask
        already_outside_tracker = (~inside_tracker) & mask
        
        # Separate neutral and charged particles (among those needing propagation)
        no_bfield = 1 if torch.abs(self.bz) < 1.0e-9 else 0
        neutral_mask = (no_bfield*(torch.ones_like(q, dtype=torch.bool)) + (1-no_bfield)*(torch.abs(q) < 1.0e-9)) & needs_propagation
        charged_mask = (~(torch.abs(q) < 1.0e-9)) & needs_propagation
        
        # ==================== NEUTRAL PARTICLE PROPAGATION ====================
        particles = self._propagate_neutral(
            particles, neutral_mask,
            x, y, z, px, py, pz, pt, e
        )
    
        # ==================== CHARGED PARTICLE PROPAGATION ====================
        particles = self._propagate_charged(
            particles, charged_mask,
            x, y, z, px, py, pz, pt, e, q
        )
        
        # ==================== COMPUTE ETA FOR "ALREADY OUTSIDE" PARTICLES ====================
        # Particles that were already outside tracker don't go through propagation
        # but still need position-based Eta computed at their current position
        x_out = particles[already_outside_tracker, 11] * 1.0e-2  # cm to m
        y_out = particles[already_outside_tracker, 12] * 1.0e-2
        z_out = particles[already_outside_tracker, 13] * 1.0e-2
        r_xy_out = torch.sqrt(x_out**2 + y_out**2)
        eta_out = torch.asinh(z_out / (r_xy_out + 1e-10))
        particles[already_outside_tracker, 8] = eta_out
    
        # Filter out particles that didn't reach detector (r_t == 0)
        # In _propagate_charged, invalid particles have position set to zero
        # But DON'T filter particles that were already outside (they should keep their positions)
        final_r = torch.sqrt(particles[:, 11]**2 + particles[:, 12]**2) * 1.0e-3
        reached_detector = (final_r > 1.0e-6) | already_outside_tracker
        mask = mask & reached_detector
        
        # Update the mask column
        particles[:, 15] = mask.float()
        
        # Separate by type using PID and charge
        pid = particles[:, 0]
        q_final = particles[:, 2]
        
        abs_pid = torch.abs(pid)
        is_charged = torch.abs(q_final) > 1.0e-9
        
        # Create type-specific masks (all relative to full particle array)
        electron_mask = mask & (abs_pid == 11) & is_charged
        muon_mask = mask & (abs_pid == 13) & is_charged
        charged_hadron_mask = mask & is_charged & ~electron_mask & ~muon_mask
        neutral_mask_out = mask & ~is_charged
        
        # Create outputs with type-specific masks
        result = {
            'ParticleAfterProp': particles.clone(),  # All particles with updated mask
            'ChargedHadron': self._apply_type_mask(particles, charged_hadron_mask),
            'Electron': self._apply_type_mask(particles, electron_mask),
            'Muon': self._apply_type_mask(particles, muon_mask),
            'NeutralParticleAfterProp': self._apply_type_mask(particles, neutral_mask_out)
        }
        
        return result
    
    def _apply_type_mask(self, particles, type_mask):
        """Apply a particle-type specific mask to the output."""
        output = particles.clone()
        output[:, 15] = type_mask.float()
        return output
    
    def _propagate_neutral(self, particles, mask, x, y, z, px, py, pz, pt, e):
        """
        Propagate neutral particles in straight lines.
        Updates positions in-place for particles where mask=True.
        
        Solves: pt^2*t^2 + 2*(px*x + py*y)*t - (radius^2 - x^2 - y^2) = 0
        for time t to reach detector cylinder.
        """
        if not mask.any():
            return particles
        
        # Extract neutral particle data
        x_n = x[mask]
        y_n = y[mask]
        z_n = z[mask]
        px_n = px[mask]
        py_n = py[mask]
        pz_n = pz[mask]
        pt_n = pt[mask]
        e_n = e[mask]
        
        pt2_n = pt_n**2
        
        # Time to reach cylinder sides (solve quadratic)
        tmp = px_n * y_n - py_n * x_n
        discriminant = pt2_n * self.radius2 - tmp**2
        discriminant = torch.clamp(discriminant, min=0.0)  # Ensure non-negative
        
        t_r = (torch.sqrt(discriminant) - px_n * x_n - py_n * y_n) / (pt2_n + 1e-10)
        
        # Time to reach cylinder ends
        t_z = torch.where(
            torch.abs(pz_n) > 1e-10,
            (torch.sign(pz_n) * self.half_length - z_n) / pz_n,
            torch.tensor(1.0e99, device=self.device)
        )
        
        # Take minimum time
        t = torch.min(t_r, t_z)
        
        # Compute final position
        x_t = x_n + px_n * t
        y_t = y_n + py_n * t
        z_t = z_n + pz_n * t
        
        # Path length
        dx = x_t - x_n
        dy = y_t - y_n
        dz = z_t - z_n
        path_length = torch.sqrt(dx**2 + dy**2 + dz**2)
        
        # Compute position-based eta at final position (EtaOuter)
        r_t_xy = torch.sqrt(x_t**2 + y_t**2)
        eta_outer = torch.asinh(z_t / (r_t_xy + 1e-10))
        
        # Update positions and EtaOuter in particles (convert m back to mm)
        particles[mask, 8] = eta_outer  # EtaOuter (position eta at final position)
        particles[mask, 11] = x_t * 1.0e3  # X
        particles[mask, 12] = y_t * 1.0e3  # Y
        particles[mask, 13] = z_t * 1.0e3  # Z
        particles[mask, 10] = particles[mask, 10] + t * e_n * 1.0e3  # T (time in mm/c)
        
        # Store path length (could be stored in a new column if needed)
        # For now we don't have a dedicated column for L in the 16-column format
        
        return particles
    
    def _propagate_charged(self, particles, mask, x, y, z, px, py, pz, pt, e, q):
        """
        Propagate charged particles in helical paths through magnetic field.
        Updates positions in-place for particles where mask=True.
        
        This implements the helix propagation from C++ Delphes ParticlePropagator.
        """
        if not mask.any():
            return particles
        
        # Extract charged particle data
        x_c = x[mask]
        y_c = y[mask]
        z_c = z[mask]
        px_c = px[mask]
        py_c = py[mask]
        pz_c = pz[mask]
        pt_c = pt[mask]
        e_c = e[mask]
        q_c = q[mask]
        
        # 1. Calculate helix parameters
        # gammam = E / c^2 (in eV/c^2)
        gammam = e_c * 1.0e9 / (self.c_light * self.c_light)
        
        # Gyration frequency: omega = q * Bz / gammam (in rad/s)
        omega = q_c * self.bz / gammam
        
        # Helix radius: r = pt / (q * Bz) * c (in meters)
        r = pt_c / (q_c * self.bz) * 1.0e9 / self.c_light
        
        # Initial phi angle
        phi_0 = torch.atan2(py_c, px_c)
        
        # 2. Helix center coordinates
        x_center = x_c + r * torch.sin(phi_0)
        y_center = y_c - r * torch.cos(phi_0)
        r_center = torch.sqrt(x_center**2 + y_center**2)
        
        # 3. Calculate propagation time
        # Velocity along z
        vz = pz_c * self.c_light / e_c
        
        # Time to reach z boundaries
        t_z = torch.where(
            torch.abs(vz) > 1e-10,
            (torch.sign(pz_c) * self.half_length - z_c) / vz,
            torch.tensor(1.0e99, device=self.device)
        )
        
        # Check if helix crosses cylinder sides
        crosses_sides = (r_center + torch.abs(r)) >= self.radius
        
        # Time to reach cylinder sides (for helices that cross)
        # Use law of cosines to find angle
        cos_arg = (r**2 + r_center**2 - self.radius**2) / (2.0 * torch.abs(r) * r_center + 1e-10)
        cos_arg = torch.clamp(cos_arg, -1.0, 1.0)
        alpha = torch.acos(cos_arg)
        
        # Time of closest approach
        td = (phi_0 + torch.atan2(x_center, y_center)) / omega
        
        # Remove modulo pi ambiguities
        pio = torch.abs(torch.pi / omega)
        td = torch.where(
            torch.abs(td) > 0.5 * pio,
            td - torch.sign(td) * pio,
            td
        )
        
        t_r = td + torch.abs(alpha / omega)
        
        # Choose minimum time
        t = torch.where(crosses_sides, torch.min(t_r, t_z), t_z)
        
        # 4. Calculate final position
        phi_t = phi_0 - omega * t
        x_t = x_center - r * torch.sin(phi_t)
        y_t = y_center + r * torch.cos(phi_t)
        z_t = z_c + vz * t
        r_t = torch.sqrt(x_t**2 + y_t**2)
        
        # Path length
        path_length = torch.abs(t) * torch.sqrt(vz**2 + (r * omega)**2)
        
        # Only update particles that successfully reached detector
        valid = r_t > 0.0
        
        # Calculate track parameters at closest approach
        # IMPORTANT: Momentum is updated at closest approach (phid), NOT at final position (phi_t)
        # This matches C++ Delphes behavior (line 295-296 in ParticlePropagator.cc)
        phid = phi_0 - omega * td
        xd = x_center - r * torch.sin(phid)
        yd = y_center + r * torch.cos(phid)
        zd = z_c + vz * td
        
        # Update momentum direction at CLOSEST APPROACH (not final position)
        px_d = pt_c * torch.cos(phid)
        py_d = pt_c * torch.sin(phid)
        # pz unchanged
        
        # Compute position-based eta at final position (EtaOuter)
        r_t_xy = torch.sqrt(x_t[valid]**2 + y_t[valid]**2)
        eta_outer = torch.asinh(z_t[valid] / (r_t_xy + 1e-10))
        
        # Update particles for valid propagations
        # We need to map from masked indices to full particle array
        mask_indices = torch.where(mask)[0]
        valid_indices = mask_indices[valid]
        
        particles[valid_indices, 4] = px_d[valid]  # Px (at closest approach)
        particles[valid_indices, 5] = py_d[valid]  # Py (at closest approach)
        particles[valid_indices, 8] = eta_outer  # EtaOuter (position eta at final position)
        particles[valid_indices, 9] = phid[valid]  # Phi (at closest approach)
        particles[valid_indices, 11] = x_t[valid] * 1.0e3  # X (m to mm) - final position
        particles[valid_indices, 12] = y_t[valid] * 1.0e3  # Y (m to mm) - final position
        particles[valid_indices, 13] = z_t[valid] * 1.0e3  # Z (m to mm) - final position
        particles[valid_indices, 10] = particles[valid_indices, 10] + t[valid] * self.c_light * 1.0e3  # T
        
        # Mark invalid particles by setting position to zero
        invalid_indices = mask_indices[~valid]
        particles[invalid_indices, 11] = 0.0
        particles[invalid_indices, 12] = 0.0
        particles[invalid_indices, 13] = 0.0
        
        return particles


# Example usage and testing
if __name__ == "__main__":
    print("Testing Delphes ParticlePropagator PyTorch Module\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create some example particles (GenParticle format)
    n_particles = 10
    
    particles = torch.zeros((n_particles, 15), dtype=torch.float64)
    
    # Generate random particles at origin with various momenta
    pt_values = torch.rand(n_particles) * 10 + 0.5  # 0.5 to 10.5 GeV
    eta_values = torch.rand(n_particles) * 4 - 2  # -2 to 2
    phi_values = torch.rand(n_particles) * 2 * np.pi - np.pi
    
    # Mix of charged and neutral particles
    charges = torch.tensor([1, -1, 0, 1, -1, 0, 1, 1, 0, -1], dtype=torch.float64)
    pids = torch.tensor([211, -211, 22, 11, -11, 2112, 13, 2212, 111, -13], dtype=torch.float64)
    
    # Fill particle data
    particles[:, 0] = pids  # PID
    particles[:, 1] = 1  # Status
    particles[:, 2] = charges  # Charge
    
    # Momentum
    particles[:, 4] = pt_values * torch.cos(phi_values)  # Px
    particles[:, 5] = pt_values * torch.sin(phi_values)  # Py
    particles[:, 6] = pt_values * torch.sinh(eta_values)  # Pz
    particles[:, 7] = pt_values  # PT
    particles[:, 8] = eta_values  # Eta
    particles[:, 9] = phi_values  # Phi
    
    # Energy (approximate for massless)
    p = torch.sqrt(particles[:, 4]**2 + particles[:, 5]**2 + particles[:, 6]**2)
    particles[:, 3] = p  # E
    
    # Position (at origin, mm)
    particles[:, 11] = 0.0  # X
    particles[:, 12] = 0.0  # Y
    particles[:, 13] = 0.0  # Z
    particles[:, 10] = 0.0  # T
    
    # Mass
    particles[:, 14] = 0.140  # Pion mass
    
    print("Input particles:")
    print(f"Shape: {particles.shape}")
    print(f"Number of particles: {n_particles}")
    print(f"Charges: {charges.numpy()}")
    print(f"PIDs: {pids.numpy()}\n")
    
    # Create propagator module (CMS-like parameters)
    propagator = ParticlePropagator(
        radius=1.29,        # CMS tracker radius
        half_length=3.0,    # CMS tracker half-length
        bz=3.8,            # CMS magnetic field
        device='cpu'
    )
    
    # Propagate particles
    result = propagator(particles)
    
    print("="*70)
    print("Propagation Results:")
    print("="*70)

    print(f"\nAll propagated particles: {result['ParticleAfterProp'].shape[0]}")
    print(f"Charged hadrons: {result['ChargedHadron'].shape[0]}")
    print(f"Electrons: {result['Electron'].shape[0]}")
    print(f"Muons: {result['Muon'].shape[0]}")
    print(f"Neutrals: {result['NeutralParticleAfterProp'].shape[0]}")

    if result['ParticleAfterProp'].shape[0] > 0:
        print("\nFinal positions (first 5):")
        print("X (mm):", result['all'][:5, 11].numpy())
        print("Y (mm):", result['all'][:5, 12].numpy())
        print("Z (mm):", result['all'][:5, 13].numpy())
        
        # Calculate final radius
        final_r = torch.sqrt(result['all'][:, 11]**2 + result['all'][:, 12]**2)
        print(f"\nFinal radii (mm): min={final_r.min():.1f}, max={final_r.max():.1f}")
    
    print("\n✓ ParticlePropagator test completed successfully!")
