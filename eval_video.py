#!/usr/bin/env python
def eval_davis_result(results_path, davis_path):
    import os
    import sys
    from time import time
    import argparse

    import numpy as np
    import pandas as pd
    from davis2017.evaluation import DAVISEvaluation

    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis_path', type=str, help='Path to the DAVIS folder containing the JPEGImages, Annotations, '
                                                    'ImageSets, Annotations_unsupervised folders')
    parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
    parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised',
                        choices=['semi-supervised', 'unsupervised'])
    parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders', default='')
    args, _ = parser.parse_known_args()
    csv_name_global = f'global_results-{args.set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

    # Check if the method has been evaluated before, if so read the results, otherwise compute the results
    args.results_path = results_path
    args.davis_path = davis_path
    csv_path = args.results_path.replace("result", "result_csv")
    os.makedirs(csv_path, exist_ok=True)
    csv_name_global_path = os.path.join(csv_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(csv_path, csv_name_per_sequence)
   
    print(f'Evaluating sequences for the {args.task} task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=args.davis_path, task=args.task, gt_set=args.set)
    metrics_res = dataset_eval.evaluate(args.results_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                    np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))

