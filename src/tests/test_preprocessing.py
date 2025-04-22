"""
Testcases for the preprocessing
"""

import pandas as pd
import io
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.run_preprocessing import run_preprocessing
from instancespace.stages.preprocessing import PreprocessingOutput

from instancespace.stages.prelim import PrelimStage, PrelimInput
from instancespace.stages.sifted import SiftedStage, SiftedInput
from instancespace.data.options import SelvarsOptions, PrelimOptions, SiftedOptions, ParallelOptions


# @pytest.fixture
# def metadata_path():
#     return "src/tests/metadata.csv"

def test_run_preprocessing_returns_output(metadata_path):
    # Run the preprocessing with default feature and algorithm selection
    output = run_preprocessing(metadata_path)

    return output


preprocessing_output = test_run_preprocessing_returns_output("metadata.csv")
print(preprocessing_output)

feats = ['feature_Max_Normalized_Entropy_attributes', 'feature_Normalized_Entropy_Class_Attribute', 'feature_Mean_Mutual_Information_Attribute_Class', 'feature_ErrorRate_Decision_Node', 'feature_WeightedDist_StdDev', 'feature_Max_Feature_Efficiency_F3', 'feature_Collective_Feature_Efficiency_F4', 'feature_Training_Error_Linear_Classifier_L2', 'feature_Fraction_Points_Class_Boundary_N1', 'feature_Nonlinearity_Nearest_Neighbor_Classifier_N4']
algos = ['algo_NB', 'algo_LDA', 'algo_QDA', 'algo_CART', 'algo_J48', 'algo_KNN', 'algo_L_SVM', 'algo_poly_SVM', 'algo_RBF_SVM', 'algo_RandF']

selvars_opts = SelvarsOptions(
    feats=feats,
    algos=algos,
    small_scale_flag=False,
    small_scale=0.1,
    file_idx_flag=False,
    file_idx="",
    selvars_type="manual",
    min_distance=0.0,
    density_flag=False
)



prelim_opts = PrelimOptions(
    max_perf=0,
    abs_perf=1,
    epsilon=0.2000,
    beta_threshold=0.5500,
    bound=1,
    norm=1
)

prelim_in = PrelimInput(
    x               = preprocessing_output.x,
    y               = preprocessing_output.y,
    x_raw           = preprocessing_output.x_raw,
    y_raw           = preprocessing_output.y_raw,
    s               = preprocessing_output.s,
    inst_labels     = preprocessing_output.inst_labels,
    prelim_options  = prelim_opts,
    selvars_options = selvars_opts,
)


prelim_out = PrelimStage._run(prelim_in)


print('------')
print(type(prelim_out))
print(prelim_out)

selvars_opts_sifted = SelvarsOptions(
    feats=None,          # None = use every feature that arrives from Prelim
    algos=None,          # None = keep every algorithm
    small_scale_flag=False,
    small_scale=0.1,
    file_idx_flag=False,
    file_idx="",
    selvars_type="manual",
    min_distance=0.0,
    density_flag=False,
)

sifted_opts = SiftedOptions(
    rho=0.15,               # min |ρ| to keep a feature after corr filter
    k=5,                    # #clusters in correlation clustering
    max_iter=300,
    replicates=10,
    num_generations=50,
    num_parents_mating=10,
    sol_per_pop=20,
    parent_selection_type="tournament",
    k_tournament=3,
    keep_elitism=2,
    crossover_type="single_point",
    cross_over_probability=0.9,
    mutation_type="random",
    mutation_probability=0.2,
    stop_criteria="saturate_5",
    flag=False,
    n_trees=100
)

parallel_opts = ParallelOptions(flag=False, n_cores=4)

feat_labels = preprocessing_output.feat_labels
x          = prelim_out.x
y          = prelim_out.y
y_bin      = prelim_out.y_bin
x_raw      = prelim_out.x_raw
y_raw      = prelim_out.y_raw
beta       = prelim_out.beta
num_good   = prelim_out.num_good_algos
y_best     = prelim_out.y_best
p_vals     = prelim_out.p
inst_lbls  = prelim_out.instlabels
s_series   = prelim_out.s             # could be None
data_dense = prelim_out.data_dense 


sifted_in = SiftedInput(
    x=x,
    y=y,
    y_bin=y_bin,
    x_raw=x_raw,
    y_raw=y_raw,
    beta=beta,
    num_good_algos=num_good,
    y_best=y_best,
    p=p_vals,
    inst_labels=inst_lbls,
    feat_labels=list(feat_labels),   # ensure it’s a list, not a numpy array
    s=s_series,
    sifted_options=sifted_opts,
    selvars_options=selvars_opts_sifted,
    data_dense=data_dense,
    parallel_options=parallel_opts,
)

print('------')

sifted_out = SiftedStage._run(sifted_in)   # or SiftedStage.sifted(**vars)

