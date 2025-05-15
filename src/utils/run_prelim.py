from instancespace.data.options import PrelimOptions, SelvarsOptions
from instancespace.stages.prelim import PrelimStage


def run_prelim(PreprocessingOutput):

    selvars_opts = SelvarsOptions(
        feats=feats,
        algos=algos,
        small_scale_flag=False,
        small_scale=0.1,
        file_idx_flag=False,
        file_idx=None,
        selvars_type="manual",
        min_distance=0.0,
        density_flag=False,
    )

    prelim_opts = PrelimOptions(
        max_perf=0,
        abs_perf=1,
        epsilon=0.2000,
        beta_threshold=0.5500,
        bound=1,
        norm=1,
    )

    prelim_in = PrelimInput(
        x=PreprocessingOutput.x,
        y=PreprocessingOutput.y,
        x_raw=PreprocessingOutput.x_raw,
        y_raw=PreprocessingOutput.y_raw,
        s=PreprocessingOutput.s,
        inst_labels=PreprocessingOutput.inst_labels,
        prelim_options=prelim_opts,
        selvars_options=selvars_opts,
    )

    prelim = PrelimStage(
        x=PreprocessingOutput.x,
        y=PreprocessingOutput.y,
        x_raw=PreprocessingOutput.x_raw,
        y_raw=PreprocessingOutput.y_raw,
        s=PreprocessingOutput.s,
        inst_labels=PreprocessingOutput.inst_labels,
        prelim_options=prelim_opts,
        selvars_options=selvars_opts,
    )

    return prelim._run(prelim_in)
