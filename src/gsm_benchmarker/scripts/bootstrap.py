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

    parser.add_argument('--log-level', type=int, default=logging.INFO)
    return parser


def main():
    pargs = make_parser().parse_args()
    install_colored_logger(level=pargs.log_level)

    output_folder = pargs.output_path or pargs.data_path.parent/"bootstrap"
    setup_log_file_handler(output_folder / 'logs')

    logger.info(f"Outputs will be saved to folder: {output_folder}")
    output_filename = (pargs.output_filename or f"boot{pargs.n_boot}") + ".csv"
    output_path = output_folder/output_filename

    logger.info(f"Loading data from {pargs.data_path}")
    mres = MultiVariantMultiModelResultsAnalyser(pargs.data_path)
    logger.debug("Data loaded")

    if pargs.save_checkpoints:
        checkpoint_filename = (pargs.checkpoint_filename or f"boot{pargs.n_boot}_checkpoints") + ".pkl"
        checkpoint_path = output_folder / checkpoint_filename
        logger.info(f"Checkpoint path is: {checkpoint_path}")

    else:
        checkpoint_path = None

    glmm, data_df = mres.prep_variant_effect(variant=pargs.variant, metric=pargs.metric)

    bs_results = glmm.run_bootstrap(
        data_df,
        n_boot=pargs.n_boot,
        checkpoint_path=checkpoint_path,
        ignore_previous_checkpoints=pargs.ignore_previous_checkpoints
    )

    logger.info("Adding single GLMM estimates (non-bootstrapped) for comparison")
    original_results, original_info = mres.analyse_variant_effect(variant=pargs.variant, metric=pargs.metric)

    summary_df = glmm.summarise_bootstrap_results(bs_results, original_results, original_info)
    summary_df.to_csv(output_path)
    logger.info(f"Bootstrap summary saved to {output_path}")

    print("Bootstrap summary:")
    print(summary_df)


if __name__ == '__main__':
    main()
