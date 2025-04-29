from instancespace.stages.pilot import PilotStage, PilotInput, PilotOutput
from instancespace.data.options import PilotOptions
from utils.cache_utils import load_from_cache
from typing import Optional


def run_pilot_from_preprocessed(use_algo_index: int = 0, analytic: bool = True) -> Optional[PilotOutput]:
    """from instancespace.stages.pilot import PilotOutput

def run_pilot_from_preprocessed(...) -> Optional[PilotOutput]:
    Run the PILOT stage using cached preprocessing output.

    Parameters
    ----------
    use_algo_index : int
        Index of algorithm column to use as projection target.
    analytic : bool
        Whether to use analytic (PCA-like) projection or numerical optimization.

    Returns
    -------
    PilotOutput
        The output of the PILOT stage.
    """
    # Step 1: Load preprocessed data from cache
    output = load_from_cache("preprocessing_output.pkl")

    # Step 2: Extract features (x), selected algorithm performance (y), and feature labels
    x = output.x
    y = output.y_raw[:, use_algo_index].reshape(-1, 1)
    feat_labels = output.feat_labels

    # Step 3: Set pilot options
    pilot_options = PilotOptions(
        x0=None,
        alpha=None,
        analytic=analytic,
        n_tries=10
    )

    # Step 4: Build and run PILOT
    pilot_input = PilotInput(
        x=x,
        y=y,
        feat_labels=feat_labels,
        pilot_options=pilot_options
    )

    return PilotStage._run(pilot_input)
