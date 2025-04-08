from instancespace.data.metadata import from_csv_file
from instancespace.data.options import SelvarsOptions
from instancespace.stages.preprocessing import PreprocessingInput

def load_metadata_and_create_input(file_path: str, selvars: SelvarsOptions) -> PreprocessingInput:
    """
    Load metadata.csv and convert it into a PreprocessingInput object.

    Parameters
    ----------
    file_path : str
        Path to the metadata.csv file (can be relative or absolute).
    selvars : SelvarsOptions
        Configuration for selecting specific features or algorithms.

    Returns
    -------
    PreprocessingInput
        An object ready to be passed into the Preprocessing stage.
    """
    metadata = from_csv_file(file_path)
    if metadata is None:
        raise ValueError(f"Failed to load metadata from {file_path}.")

    return PreprocessingInput(
        feature_names=metadata.feature_names,
        algorithm_names=metadata.algorithm_names,
        instance_labels=metadata.instance_labels,
        instance_sources=metadata.instance_sources,
        features=metadata.features,
        algorithms=metadata.algorithms,
        selvars_options=selvars,
    )
