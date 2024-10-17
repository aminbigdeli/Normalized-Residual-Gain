import os
from collections import defaultdict
import pandas as pd
import argparse

def load_qrels(qrels_file):
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            topic, _, doc_id, relevance = line.strip().split()
            qrels[topic][doc_id] = int(relevance)
    return qrels

def load_run(run_dir, run_file):
    runs = defaultdict(list)
    with open(os.path.join(run_dir, run_file), 'r') as f:
        for line in f:
            topic, _, doc_id, rank, score, _ = line.strip().split()
            runs[topic].append((doc_id, int(rank)))

    for topic in runs:
        runs[topic].sort(key=lambda x: x[1])
        runs[topic] = [doc[0] for doc in runs[topic]]
    return runs

def calculate_document_rarity(all_runs, k):
    doc_counts = defaultdict(int)
    for run in all_runs.values():
        for topic_docs in run.values():
            for doc in topic_docs[:k]:
                doc_counts[doc] += 1
    
    total_systems = len(all_runs)
    rarity = {doc: 1 - (count / total_systems) for doc, count in doc_counts.items()}
    return rarity

def precision_at_k_rareness(retrieved_docs, relevant_docs, rarity, alpha, k, qrels, topic):
    score = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            relevance = 1 if qrels[topic][doc] >= 1 else 0
            score += relevance * (1 + alpha * rarity[doc])
    return score / k

def average_precision_rareness(retrieved_docs, relevant_docs, rarity, alpha, k=10):
    num_relevant = len(relevant_docs)
    if num_relevant == 0:
        return 0.0
    score = 0
    relevant_retrieved = 0
    for rank, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            relevant_retrieved += 1
            score += precision_at_k_rareness(retrieved_docs, relevant_docs, rarity, alpha, rank + 1)
    return score / min(num_relevant, k)

def main(run_dir, qrels_file, result_dir, alpha, k):
    run_files = [f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f))]
    all_runs = {run_file: load_run(run_dir, run_file) for run_file in run_files}

    qrels = load_qrels(qrels_file)

    filtered_qrels = {}
    for topic, docs in qrels.items():
        filtered_qrels[topic] = {doc: rel for doc, rel in docs.items() if rel >= 1}

    rarity = calculate_document_rarity(all_runs, k)
    scores = []
    for run_file, runs in all_runs.items():
        precisions = []
        for topic, retrieved_docs in runs.items():
            if topic not in filtered_qrels:
                continue
            relevant_docs = [doc for doc in retrieved_docs if filtered_qrels.get(topic, {}).get(doc, 0) >= 1]
            precision_rareness = precision_at_k_rareness(retrieved_docs, relevant_docs, rarity, alpha, k, qrels=filtered_qrels, topic=topic)
            precisions.append(precision_rareness)
        if precisions:
            mean_precision = sum(precisions) / len(precisions)
            scores.append((run_file, mean_precision))

    scores_df = pd.DataFrame(scores, columns=['run', f'rareness-based-precision-{k}'])
    scores_df.to_csv(os.path.join(result_dir, f'rareness_based_precision_{k}.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate rareness-based precision at K for TREC runs.")
    parser.add_argument('--run_dir', type=str, required=True, help="Directory containing run files.")
    parser.add_argument('--qrels_file', type=str, required=True, help="Qrels file path.")
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save the result CSV file.")
    parser.add_argument('--alpha', type=float, required=True, help="Alpha value for rareness adjustment.")
    parser.add_argument('--k', type=int, required=True, help="Cut-off value for precision at K.")

    args = parser.parse_args()
    
    main(args.run_dir, args.qrels_file, args.result_dir, args.alpha, args.k)
