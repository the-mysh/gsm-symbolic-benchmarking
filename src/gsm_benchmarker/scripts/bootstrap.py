from argparse import ArgumentParser
from pathlib import Path
import logging

from gsm_benchmarker.results_analyser import MultiVariantMultiModelResultsAnalyser
from gsm_benchmarker.utils.logging_setup import install_colored_logger, setup_log_file_handler

logger = logging.getLogger(__file__)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('data_path', type=Path)

    parser.add_argument('--output-path', type=Path, default=None)
    parser.add_argument('--output-filename', type=str, default=None)
    parser.add_argument('--save-checkpoints', action='store_true', default=False)
    parser.add_argument('--checkpoint-filename', type=str, default=None)
    parser.add_argument('--ignore-previous-checkpoints', action='store_true', default=False)

    parser.add_argument('--n-boot', type=int, default=500)
    parser.add_argument('--variant', type=str, default='main')
    parser.add_argument('--metric', type=str, default='correct')

    parser.add_argument('--glmm-id', type=str, choices=['1', '2'], required=True)

    parser.add_argument('--log-level', type=int, default=logging.INFO)
    return parser


def make_names(n_boot: int, glmm_id: str):
    ext = "pkl"
    run_info = f"n_boot_{n_boot}__glmm{glmm_id}"
    return f"{run_info}.{ext}", f"{run_info}__wald.{ext}", f"{run_info}__checkpoints.{ext}"


def main():
    pargs = make_parser().parse_args()
    install_colored_logger(level=pargs.log_level)

    output_folder = pargs.output_path or pargs.data_path.parent/"bootstrap"
    setup_log_file_handler(output_folder / 'logs')

    output_filename_bootstrap, output_filename_wald, checkpoints_filename = make_names(pargs.n_boot, pargs.glmm_id)

    logger.info(f"Outputs will be saved to folder: {output_folder}")
    output_path_bootstrap = output_folder/output_filename_bootstrap
    output_path_wald = output_folder/output_filename_wald
    checkpoints_path = output_folder/checkpoints_filename

    logger.info(f"Loading data from {pargs.data_path}")
    mres = MultiVariantMultiModelResultsAnalyser(pargs.data_path)
    logger.debug("Data loaded")

    if pargs.save_checkpoints:
        logger.info(f"Checkpoint path is: {checkpoints_path}")
    else:
        checkpoints_path = None

    prep_func = getattr(mres, f"prep_glmm{pargs.glmm_id}")
    glmm, data_df = prep_func(variant=pargs.variant, metric=pargs.metric)

    bs_results = glmm.run_bootstrap(
        data_df,
        n_boot=pargs.n_boot,
        checkpoint_path=checkpoints_path,
        ignore_previous_checkpoints=pargs.ignore_previous_checkpoints
    )

    summary_df = glmm.summarise_bootstrap_results(bs_results)
    summary_df.to_pickle(output_path_bootstrap)
    logger.info(f"Bootstrap summary saved to {output_path_bootstrap}")

    logger.info("Adding single GLMM estimates (non-bootstrapped) for comparison")
    original_results, original_info = glmm.run(data_df)
    wald_summary_df = glmm.summarise_wald_results(original_results, original_info)
    wald_summary_df.to_pickle(output_path_wald)
    logger.info(f"Wald summary saved to {output_path_wald}")

    print("Bootstrap summary:")
    print(summary_df)


if __name__ == '__main__':
    main()
