import tempfile
from typing import IO, Union

from instancespace.data.metadata import from_csv_file
from instancespace.data.options import SelvarsOptions
from instancespace.stages.preprocessing import PreprocessingInput


def load_metadata_and_create_input(
    file_source: Union[str, IO], selvars: SelvarsOptions
) -> PreprocessingInput:
    """
    Load metadata and convert it into a PreprocessingInput object.

    Parameters
    ----------
    file_source : Union[str, IO]
        File path as string or uploaded file-like object (e.g., from Streamlit uploader).
    selvars : SelvarsOptions
        Configuration for selecting specific features or algorithms.

    Returns
    -------
    PreprocessingInput
        An object ready to be passed into the Preprocessing stage.
    """
    if isinstance(file_source, str):
        metadata = from_csv_file(file_source)
    else:
        file_source.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(file_source.read())
            tmp_path = tmp.name
        metadata = from_csv_file(tmp_path)

    if metadata is None:
        raise ValueError("Metadata loading failed.")

    return PreprocessingInput(
        feature_names=metadata.feature_names,
        algorithm_names=metadata.algorithm_names,
        instance_labels=metadata.instance_labels,
        instance_sources=metadata.instance_sources,
        features=metadata.features,
        algorithms=metadata.algorithms,
        selvars_options=selvars,
    )
