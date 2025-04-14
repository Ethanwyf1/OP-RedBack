from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.data.options import SelvarsOptions
from utils.data_loader import load_metadata_and_create_input
from typing import Union, IO


def run_preprocessing(metadata_source: Union[str, IO], feats: list[str] = None, algos: list[str] = None):
    """
    Run the preprocessing stage of ISA.

    Parameters
    ----------
    metadata_path : str
        Path to the metadata CSV file.
    feats : list[str], optional
        List of feature names to include (default is None, meaning use all).
    algos : list[str], optional
        List of algorithm names to include (default is None, meaning use all).

    Returns
    -------
    PreprocessingOutput
        The output from the Preprocessing stage.
    """
    selvars = SelvarsOptions(
        feats=feats,
        algos=algos,
        small_scale_flag=False,
        small_scale=0.1,
        file_idx_flag=False,
        file_idx=None,
        selvars_type="manual",
        min_distance=0.0,
        density_flag=False
    )

    input_data = load_metadata_and_create_input(metadata_source, selvars)

    preprocessing = PreprocessingStage(
        feature_names=input_data.feature_names,
        algorithm_names=input_data.algorithm_names,
        instance_labels=input_data.instance_labels,
        instance_sources=input_data.instance_sources,
        features=input_data.features,
        algorithms=input_data.algorithms,
        selvars=selvars,
    )

    return preprocessing._run(input_data)
