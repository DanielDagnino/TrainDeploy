#!/usr/bin/env python
"""
Some notes about the format.

Define annotations_pairs and subset subset_pairs:
    Format: Accepted
    score_fill: Accepted

Define annotations_pairs and subset subset_files:
    Format: Accepted
    score_fill: Accepted

Define annotations_cluster and subset subset_files:
    Format: Accepted
    score_fill: Accepted


Define annotations_cluster and subset subset_cluster:
    Format: Accepted
    score_fill: Not accepted. An error will raise. It does NOT check for any inconsistency in the results file.
    It is the user responsibility to provide a complete results file. If it is not possible, then,
    use one of the options above.


Define annotations_cluster and subset subset_pairs:
Define annotations_pairs and subset subset_cluster:
    Format: Not accepted. An error will raise.

"""
import argparse
import json
import os
import sys
from warnings import warn

import numpy as np

from is_ai.is_ai_song.evaluate.utils import compute_metrics


def compute_evaluation(fn_results, fn_annotations, fn_subsets, subset_group,
                       experiment_name, score_fill,
                       num_positives=None):
    """Compute the evaluation metrics."""
    # load files
    results = load_results(fn_results)
    results = sanitize_results(results)
    annotations = load_annotations(fn_annotations)
    subsets = load_subsets(fn_subsets, subset_group) if fn_subsets else [{"pairs": None}]
    if score_fill is not None and "clusters" in subsets[0]:
        warn('The value score-fill specified is not used when the subset is defined in a `cluster` form. '
             'The possible pairs are generated from the cluster, so that it typically contains more pairs that the '
             'ones needed.')

    # subset results and annotations
    y_true, y_pred, y_scores, _ = subset_data(results, annotations, subsets, fn_results, score_fill)

    # compute stats
    stats = compute_metrics(fn_results, fn_annotations, fn_subsets, y_true, y_pred, y_scores, num_positives)
    stats["experiment"] = experiment_name
    return stats


def main(args=None):
    """Compute evaluations."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('fn_results',
                        help='Path to csv with results. Format: file1(path);file2(path);score(float);pred(bool)')
    parser.add_argument('fn_annotations',
                        help='Path to csv with annotation. Format columns: id_cluster(str);file(path);match(bool)')
    parser.add_argument('fn_subsets', nargs="+",
                        help='Path to csv with subsets. Format: Single column with the paths '
                             '(for subset type `file`, two files must be provided)')
    parser.add_argument('--score-fill', type=float, default=None,
                        help='The defined paris without prediction will be filled with this value, '
                             'and as False matching.')
    parser.add_argument('--subset-group', default=None,
                        help='Limit subset only to certain group')
    parser.add_argument('--experiment-name', "-e", default="exp",
                        help='Name the experiment')
    # TODO limit file to random splits / subsets
    parser.add_argument('--output-file', "-o", default="results.json",
                        help='output filename')
    args = parser.parse_args(args)
    print(args)

    _stats = compute_evaluation(args.fn_results, args.fn_annotations, args.fn_subsets,
                                args.subset_group, args.experiment_name, args.score_fill)

    # dump stats to json
    json.dump(_stats, open(os.path.expanduser(args.output_file), "w"), indent=4)
    return _stats


if __name__ == '__main__':
    main(sys.argv[1:])
