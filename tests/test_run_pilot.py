import sys
import os
import numpy as np
import pandas as pd

# ✅ 确保 Python 能找到 src/ 作为模块根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils.run_pilot import run_pilot
from instancespace.stages.pilot import PilotOutput

# 📁 测试数据位置（你现在的 cache 目录）
TEST_METADATA_PATH = os.path.join("cache", "uploaded_metadata.csv")


def test_run_pilot_structure():
    """
    Ensure run_pilot returns correct structure and content.
    """
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    result = run_pilot(metadata_source=TEST_METADATA_PATH, use_algo_index=0, analytic=True)

    # ✅ 类型校验
    assert isinstance(result, PilotOutput), "Expected result to be of type PilotOutput"

    # ✅ z: 2D 投影
    assert hasattr(result, "z"), "Missing 'z' in result"
    assert isinstance(result.z, np.ndarray), "'z' should be a numpy array"
    assert result.z.ndim == 2 and result.z.shape[1] == 2, "'z' should have shape [n_instances, 2]"

    # ✅ a: 投影矩阵
    assert hasattr(result, "a"), "Missing 'a' in result"
    assert isinstance(result.a, np.ndarray), "'a' should be a numpy array"
    assert result.a.shape[0] == 2, "'a' should have shape [2, n_features]"

    # ✅ pilot_summary: 投影结果表
    assert hasattr(result, "pilot_summary"), "Missing 'pilot_summary' in result"
    assert isinstance(result.pilot_summary, pd.DataFrame), "'pilot_summary' should be a DataFrame"

    print("✅ run_pilot structure test passed.")
