"""Variable requirements configuration for models and datasets."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from parnassus.utils.typing import VarNameTuple

if TYPE_CHECKING:
    from parnassus.configs.model import GenerativeModelConfig


@dataclass(frozen=True, slots=True)
class VariableRequirements:
    """Shared variable configuration used by both models and datasets.

    This class encapsulates the variable names required for data processing,
    eliminating circular dependencies between DatasetConfig and ModelConfig.
    """

    truth_vars_to_load: VarNameTuple
    ctxt_vars: VarNameTuple
    ctxt_global_vars: VarNameTuple

    @classmethod
    def from_model_config(cls, model_config: "GenerativeModelConfig") -> "VariableRequirements":
        """Create VariableRequirements from a GenerativeModelConfig.

        Parameters
        ----------
        model_config : GenerativeModelConfig
            The model configuration to extract variables from.

        Returns
        -------
        VariableRequirements
            A new VariableRequirements instance with variables from the model.
        """
        return cls(
            truth_vars_to_load=model_config.truth_vars_to_load,
            ctxt_vars=model_config.event_model_config.variables_config.ctxt_vars,
            ctxt_global_vars=model_config.event_model_config.variables_config.ctxt_global_vars,
        )

    @property
    def ctxt_vars_stripped(self) -> list[str]:
        """Context variables with 'truth_' prefix removed.

        Returns
        -------
        list[str]
            List of context variable names without 'truth_' prefix.
        """
        return [var.replace("truth_", "") for var in self.ctxt_vars]

    @property
    def ctxt_global_vars_stripped(self) -> list[str]:
        """Global context variables with 'truth_' prefix removed.

        Returns
        -------
        list[str]
            List of global context variable names without 'truth_' prefix.
        """
        return [var.replace("truth_", "") for var in self.ctxt_global_vars]
