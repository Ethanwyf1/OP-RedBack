# utils/run_pilot.py

from instancespace.stages.pilot import PilotStage, PilotInput, PilotOutput
from instancespace.data.options import PilotOptions
import numpy as np
from numpy.typing import NDArray
from typing import List


def run_pilot(
    x: NDArray[np.double],
    y: NDArray[np.double],
    feat_labels: List[str],
    analytic: bool = True,
    n_tries: int = 10
) -> PilotOutput:
    """
    Run the PILOT stage with the given data and configuration.

    Parameters
    ----------
    x : NDArray[np.double]
        Feature matrix (instances x features)
    y : NDArray[np.double]
        Performance vector (instances x 1) for a selected algorithm
    feat_labels : list[str]
        Feature names
    analytic : bool
        Whether to use analytic projection (PCA-like)
    n_tries : int
        Number of BFGS optimization attempts (used only in numerical mode)

    Returns
    -------
    PilotOutput
        The result of the PILOT stage
    """
    options = PilotOptions(
        x0=None,
        alpha=None,
        analytic=analytic,
        n_tries=n_tries
    )

    pilot_input = PilotInput(
        x=x,
        y=y,
        feat_labels=feat_labels,
        pilot_options=options
    )

    return PilotStage._run(pilot_input)
