import sys
sys.path.append('./src')
sys.path.append('./external')

from utils.data_loader import load_metadata_and_create_input
from instancespace.data.options import SelvarsOptions

selvars = SelvarsOptions(
    feats=None,
    algos=None,
    small_scale_flag=False,
    small_scale=0.1,
    file_idx_flag=False,
    file_idx=None,
    selvars_type="manual",
    min_distance=0.0,
    density_flag=False,
)

input_obj = load_metadata_and_create_input("data/metadata.csv", selvars)

print("✅ PreprocessingInput loaded successfully!")
print(f"Feature names (前5个): {input_obj.feature_names[:5]}")
print(f"Algorithm names (前5个): {input_obj.algorithm_names[:5]}")
print(f"Feature shape: {input_obj.features.shape}")
print(f"Algorithm shape: {input_obj.algorithms.shape}")
