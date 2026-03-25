"""Default detector configurations for TorchDelphes.

Provides pre-configured detector simulation modules that match the
standard Delphes TCL cards for CMS and ATLAS detectors.

Classes:
    CMSEnergyFlowDefault: CMS detector with 3.8T field, r=1.29m, z=3.0m
    ATLASEnergyFlowDefault: ATLAS detector with 2.0T field, r=1.15m, z=3.51m

Examples
--------
>>> from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
>>> cms = CMSEnergyFlowDefault(debug=True)
>>> results = cms(particles)
"""

from .ATLASDefault import ATLASEnergyFlowDefault
from .CMSDefault import CMSEnergyFlowDefault

__all__ = ["ATLASEnergyFlowDefault", "CMSEnergyFlowDefault"]
