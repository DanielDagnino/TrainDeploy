import argparse
import json
import sys

from path import Path

from is_ai.is_ai_song.evaluate.plotting import *


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.makedirs_p()

    dict_results = [json.load(open(_)) for _ in args.fn_results]

    # tpr_fpr curve
    fig1, ax = plt.subplots()
    for result in dict_results:
        _ = plot_tpr_fpr(
            result["tpr_fpr_curve"]["fpr"],
            result["tpr_fpr_curve"]["tpr"],
            'voice_cloner',
            ax,
            title=args.caption,
            zoom=None,
            show_grid=True)

    fig1.savefig(out_dir / 'tpr_fpr_curve.png', bbox_inches='tight', dpi=600)

    # tpr_fpr curve, log scale
    fig1, ax = plt.subplots()
    for result in dict_results:
        _ = plot_tpr_fpr(
            result["tpr_fpr_curve"]["fpr"],
            result["tpr_fpr_curve"]["tpr"],
            'voice_cloner',
            ax,
            title=args.caption,
            zoom=(0.00000096, 1.03, -0.005, 1.005),
            show_grid=True,
            log_scale=True)

    fig1.savefig(out_dir / 'tpr_fpr_curve_log.png', bbox_inches='tight', dpi=600)

    # pr curve
    fig1, ax = plt.subplots()
    for result in dict_results:
        _ = plot_pr_curve(
            result["PR_curve"]["precisions"],
            result["PR_curve"]["recalls"],
            'voice_cloner',
            ax,
            title=args.caption)
    fig1.savefig(out_dir / 'pr.png', bbox_inches='tight', dpi=600)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fn_results', nargs="+", help='File list with the results (metrics).')
    parser.add_argument('--caption', default="", help='additional caption')
    parser.add_argument('--out-dir', default=".", help='out directory')
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
