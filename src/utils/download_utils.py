import pandas as pd
import zipfile
import io
from typing import Optional

def create_stage_output_zip(
    x: pd.DataFrame,
    y: pd.DataFrame,
    instance_labels: pd.Series,
    source_labels: Optional[pd.Series],
    feature_filename: str = "features.csv",
    performance_filename: str = "performance.csv",
    include_metadata_txt: bool = True,
    metadata_description: Optional[str] = None
) -> bytes:
    """
    Creates a ZIP file in memory containing stage output data (features + performance).

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix with columns = selected features.
    y : pd.DataFrame
        Algorithm performance matrix with columns = algorithm names.
    instance_labels : pd.Series
        Row-level instance identifiers.
    source_labels : Optional[pd.Series]
        Optional instance source column.
    feature_filename : str
        CSV name for feature matrix.
    performance_filename : str
        CSV name for performance matrix.
    include_metadata_txt : bool
        Whether to include README.txt.
    metadata_description : Optional[str]
        Custom text for metadata file.

    Returns
    -------
    bytes
        Bytes representing a ZIP archive for download.
    """
    features_df = x.copy()
    performance_df = y.copy()

    features_df.insert(0, "Instance", instance_labels.values)
    performance_df.insert(0, "Instance", instance_labels.values)

    if source_labels is not None:
        features_df.insert(1, "Source", source_labels.values)
        performance_df.insert(1, "Source", source_labels.values)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(feature_filename, features_df.to_csv(index=False))
        zipf.writestr(performance_filename, performance_df.to_csv(index=False))

        if include_metadata_txt:
            metadata = metadata_description or (
                f"{feature_filename} contains the feature matrix.\n"
                f"{performance_filename} contains the algorithm performance matrix.\n"
                "Each includes 'Instance' and optional 'Source' columns.\n"
            )
            zipf.writestr("README.txt", metadata)

    return zip_buffer.getvalue()
