# utils/run_pythia.py

from typing import Union, IO
from instancespace.data.options import PythiaOptions, ParallelOptions
from instancespace.stages.pythia import PythiaStage, PythiaOutput
from utils.cache_utils import load_from_cache, cache_exists


def run_pythia(
    pythia_options: PythiaOptions = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=True,
        use_grid_search=True,
        params=None,  # Optional pre-defined hyperparameters for each algorithm
    ),
    parallel_options: ParallelOptions = ParallelOptions(
        flag=False,
        n_cores=1
    )
) -> PythiaOutput:
    """
    Run the PYTHIA stage of ISA.

    This function assumes that the necessary outputs from previous stages
    (PILOT, PRELIM, SIFTED, and PREPROCESSING) have already been cached.

    Parameters
    ----------
    pythia_options : PythiaOptions
        Configuration for SVM training, kernel type, CV folds, etc.
    parallel_options : ParallelOptions
        Settings for parallel execution.

    Returns
    -------
    PythiaOutput
        Output from PYTHIA stage including trained models and performance summary.
    """

    # --- Load required intermediate stage outputs ---
    pilot_output = load_from_cache("pilot_output.pkl")
    prelim_output = load_from_cache("prelim_output.pkl")
    sifted_output = load_from_cache("sifted_output.pkl")
    preprocessing_output = load_from_cache("preprocessing_output.pkl")

    # --- Check for any missing stage ---
    if None in [pilot_output, prelim_output, sifted_output, preprocessing_output]:
        raise ValueError("❌ Required stage outputs (PILOT, PRELIM, SIFTED, PREPROCESSING) are missing from cache.")

    # --- Extract algorithm labels ---
    algo_labels = preprocessing_output.algo_labels

    # --- Check label/data alignment ---
    if sifted_output.y.shape[1] != len(algo_labels):
        raise ValueError(
            f"❌ Dimension mismatch: sifted_output.y has {sifted_output.y.shape[1]} columns "
            f"but algo_labels has {len(algo_labels)} entries."
        )

    print(f"✅ Loaded {len(algo_labels)} algorithm labels from preprocessing.")
    print(f"✅ SIFTED output has {sifted_output.y.shape[1]} algorithm columns.")

    # --- Build stage input ---
    stage_input = PythiaStage._inputs()(
        z=pilot_output.z,
        y_raw=sifted_output.y,
        y_bin=prelim_output.y_bin,
        y_best=prelim_output.y_best,
        algo_labels=algo_labels,
        pythia_options=pythia_options,
        parallel_options=parallel_options,
    )

    # --- Run PYTHIA Stage ---
    return PythiaStage._run(stage_input)