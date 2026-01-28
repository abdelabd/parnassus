import torch

def ecal_cms_resolution(energy: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    CMS ECAL energy resolution formula.
    From delphes_card_CMS_5_0.tcl
    """
    abs_eta = torch.abs(eta)
    
    # Barrel: |eta| <= 1.5
    barrel_mask = abs_eta <= 1.5
    barrel_res = (1.0 + 0.64 * (eta**2)) * torch.sqrt(
        energy**2 * (0.008**2) + energy * (0.11**2) + (0.40**2)
    )
    
    # Endcap: 1.5 < |eta| <= 2.5
    endcap_mask = (abs_eta > 1.5) & (abs_eta <= 2.5)
    endcap_res = (2.16 + 5.6 * (abs_eta - 2.0)**2) * torch.sqrt(
        energy**2 * (0.008**2) + energy * (0.11**2) + (0.40**2)
    )
    
    # HF: 2.5 < |eta| <= 5.0
    hf_mask = (abs_eta > 2.5) & (abs_eta <= 5.0)
    hf_res = torch.sqrt(energy**2 * (0.107**2) + energy * (2.08**2))
    
    resolution = torch.zeros_like(energy)
    resolution = torch.where(barrel_mask, barrel_res, resolution)
    resolution = torch.where(endcap_mask, endcap_res, resolution)
    resolution = torch.where(hf_mask, hf_res, resolution)
    
    return resolution

def hcal_cms_resolution(energy: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    CMS HCAL energy resolution formula (placeholder).
    """
    # Simple parametrization for HCAL
    return torch.sqrt(energy**2 * 0.1**2 + energy * 0.5**2 + 1.0**2)
    