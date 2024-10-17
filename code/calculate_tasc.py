import pandas as pd
from collections import defaultdict
from typing import List, Dict
import numpy as np
import pytrec_eval
import os
import glob
import argparse

def load_run_file(run_file: str, runs_dir: str) -> Dict[str, Dict[str, float]]:
    run_data = defaultdict(dict)
    with open(os.path.join(runs_dir, run_file), 'r') as file:
        for line in file:
            query_id, _, doc_id, rank, score, _ = line.split()
            run_data[query_id][doc_id] = float(score)
    return run_data

def load_qrels_file(qrels_file: str) -> Dict[str, Dict[str, int]]:
    qrels_data = defaultdict(dict)
    with open(qrels_file, 'r') as file:
        for line in file:
            query_id, _, doc_id, relevance = line.split()
            qrels_data[query_id][doc_id] = int(relevance)
    return qrels_data

def filter_judged_queries(run_data: Dict[str, Dict[str, float]], qrels_data: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    judged_queries = set(qrels_data.keys())
    return {query_id: docs for query_id, docs in run_data.items() if query_id in judged_queries}

def calculate_ndcg(run_data: Dict[str, Dict[str, float]], qrels_data: Dict[str, Dict[str, int]], ndcg_cut: int) -> Dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_data, {f'ndcg_cut.{ndcg_cut}'})
    results = evaluator.evaluate(run_data)
    ndcg_scores = {query_id: metrics[f'ndcg_cut_{ndcg_cut}'] for query_id, metrics in results.items()}
    return ndcg_scores

def calculate_all_ndcg(run_files: List[str], qrels_data: Dict[str, Dict[str, int]], runs_dir: str, ndcg_cut: int) -> Dict[str, Dict[str, float]]:
    all_runs_ndcg = {}
    for run_file in run_files:
        run_data = load_run_file(run_file, runs_dir)
        filtered_run_data = filter_judged_queries(run_data, qrels_data)
        ndcg_scores = calculate_ndcg(filtered_run_data, qrels_data, ndcg_cut)
        all_runs_ndcg[run_file] = ndcg_scores
    return all_runs_ndcg

def calculate_max_tasc(target_run: str, all_runs_ndcg: Dict[str, Dict[str, float]], run_files: List[str]) -> float:
    target_run_ndcg = all_runs_ndcg[target_run]
    prior_runs_ndcg = [all_runs_ndcg[run_file] for run_file in run_files]
    query_tasc_scores = []

    for query_id, target_ndcg in target_run_ndcg.items():
        max_prior_ndcg = max((run.get(query_id, 0) for run in prior_runs_ndcg), default=0)
        tasc_score = (1 - max_prior_ndcg) * target_ndcg
        query_tasc_scores.append(tasc_score)

    if query_tasc_scores:
        average_tasc = np.mean(query_tasc_scores)
    else:
        average_tasc = 0.0

    return average_tasc

def main(runs_dir: str, qrels_file: str, run_group_file: str, best_runs_dir: str, result_dir: str, ndcg_cut: int) -> None:
    qrels_data = load_qrels_file(qrels_file)

    run_files = [f for f in os.listdir(runs_dir) if os.path.isfile(os.path.join(runs_dir, f))]
    all_runs_ndcg = calculate_all_ndcg(run_files, qrels_data, runs_dir, ndcg_cut)

    runs = pd.read_csv(run_group_file, names=['group', 'run_name'])
    run_group_dict = {row['run_name']: row['group'] for _, row in runs.iterrows()}

    files = glob.glob(os.path.join(best_runs_dir, "*"))
    best_runs = [os.path.basename(run_name) for run_name in files]

    runs_list = runs['run_name'].values.tolist()
    scores = []
    for _, row in runs.iterrows():
        target_run = row['run_name']
        group_target_run = run_group_dict[target_run]
        competitor_runs = [run for run in best_runs if run_group_dict[run] == group_target_run and run != target_run]

        result = calculate_max_tasc(target_run, all_runs_ndcg, competitor_runs)
        scores.append([group_target_run, target_run, result])

    scores_df = pd.DataFrame(scores, columns=['group', 'run', 'TaSC'])
    scores_df.to_csv(os.path.join(result_dir, 'tasc_scores.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate NDCG and TaSC scores for TREC runs.")
    parser.add_argument('--runs_dir', type=str, required=True, help="Directory containing run files.")
    parser.add_argument('--qrels_file', type=str, required=True, help="Qrels file path.")
    parser.add_argument('--run_group_file', type=str, required=True, help="CSV file with run group names.")
    parser.add_argument('--best_runs_dir', type=str, required=True, help="Directory containing best run files.")
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save the result CSV file.")
    parser.add_argument('--ndcg_cut', type=int, required=True, help="NDCG cut-off value.")

    args = parser.parse_args()
    
    main(args.runs_dir, args.qrels_file, args.run_group_file, args.best_runs_dir, args.result_dir, args.ndcg_cut)
