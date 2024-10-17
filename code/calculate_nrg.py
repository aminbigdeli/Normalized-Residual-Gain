import os
import multiprocessing as mp
from tqdm import tqdm
import time
import glob
import argparse
import pandas as pd
import math
from functools import partial


class NormalizedResidualGainCalculator:

    def __init__(self, dataset_name, run_directory, qrels_file, target_ranker, competitor_rankers, doc_gain_version, result_directory):
        self.dataset_name = dataset_name
        self.run_directory = run_directory
        self.qrels_file = qrels_file
        self.result_directory = result_directory
        self.queries_run_data = {}
        self.queries_qrels = {}
        self.rankers = []
        self.target_ranker = target_ranker
        self.competitor_rankers = competitor_rankers
        self.doc_gain_version = doc_gain_version

    def calculate_document_gain_v1(self, qid, doc_id):
        return self.queries_qrels.get(qid, {}).get(doc_id, 0)

    def calculate_document_gain_v2(self, qid, doc_id):
        relevance_level = self.queries_qrels.get(qid, {}).get(doc_id, 0)
        return 1 if relevance_level in [2, 3] else 0

    def calculate_document_gain_v3(self, qid, doc_id):
        relevance_level = self.queries_qrels.get(qid, {}).get(doc_id, 0)
        return (2 ** relevance_level) - 1

    def calculate_position_discount_v1(self, rank):
        return 0 if rank == 0 else 1 / math.log2(rank + 1)

    def verify_run_files(self, run_files):
        for run_file in run_files:
            with open(run_file, 'r') as file:
                unique_qids = {line.split(" ")[0] for line in file}
            break

        for run_file in run_files:
            run_name = os.path.basename(run_file).split(".")[0]
            run_data = pd.read_csv(run_file, sep=" ", names=['qid', 'Q0', 'doc_id', 'rank', 'score', 'ranker'])
            self.rankers.append(run_name)
            for qid, group in run_data.groupby(['qid']):
                qid = int(qid)
                if qid not in self.queries_qrels:
                    continue
                if qid not in self.queries_run_data:
                    self.queries_run_data[qid] = {run_name: group['doc_id'].tolist()}
                else:
                    self.queries_run_data[qid][run_name] = group['doc_id'].tolist()
        return True

    def process_qrels_file(self):
        qrels_data = pd.read_csv(self.qrels_file, sep=" ", names=['qid', 'Q0', 'doc_id', 'relevance_score'])
        for qid, group in qrels_data.groupby(['qid']):
            self.queries_qrels[qid] = dict(group[['doc_id', 'relevance_score']].values)

    def calculate_document_value(self, qid, doc_id, runs):
        doc_gain_calculators = {
            1: self.calculate_document_gain_v1,
            2: self.calculate_document_gain_v2,
            3: self.calculate_document_gain_v3
        }
        doc_gain = doc_gain_calculators[self.doc_gain_version](qid, doc_id)
        doc_exposure = 1

        for run_docs in runs.values():
            rank = run_docs.index(doc_id) + 1 if doc_id in run_docs else 0
            doc_exposure *= 1 - self.calculate_position_discount_v1(rank)

        return doc_gain * doc_exposure

    def evaluate_query(self, qid, runs, target_run):
        position_discounts = [self.calculate_position_discount_v1(rank) for rank in range(1, len(target_run) + 1)]
        return sum(self.calculate_document_value(qid, doc_id, runs) * discount for doc_id, discount in zip(target_run, position_discounts))

    def calculate_ideal_ranking(self, qid, competitors):
        qrels = self.queries_qrels[qid]
        doc_values = {doc_id: self.calculate_document_value(qid, doc_id, competitors) for doc_id in qrels}
        sorted_doc_values = sorted(doc_values.items(), key=lambda item: item[1], reverse=True)
        return sum(value * self.calculate_position_discount_v1(rank + 1) for rank, (doc_id, value) in enumerate(sorted_doc_values))

    def calculate_unique_contributions(self):
        unique_contributions = {}

        for qid, run_data in self.queries_run_data.items():
            relevant_docs = {doc_id for doc_id, relevance in self.queries_qrels.get(qid, {}).items() if relevance >= 1}
            target_docs = set(run_data.get(self.target_ranker, [])[:10]) & relevant_docs

            competitor_docs = set()
            for run_name, docs in run_data.items():
                if run_name in self.competitor_rankers:
                    competitor_docs.update(docs[:10])
            competitor_docs &= relevant_docs

            unique_docs = target_docs - competitor_docs
            unique_contributions[qid] = len(unique_docs) / 10

        total_unique_contributions = sum(unique_contributions.values())
        return unique_contributions, total_unique_contributions

    def compare_rankers(self):
        run_files = glob.glob(os.path.join(self.run_directory, "*"))
        self.process_qrels_file()
        if self.verify_run_files(run_files):
            evaluation_results = []
            ideal_results = []
            normalized_results = []
            unique_contributions = []

            for run_file in run_files:
                run_name = os.path.basename(run_file).split(".")[0]
                if self.target_ranker and self.target_ranker != run_name:
                    continue
                
                total_evaluation_score, total_normalized_score, total_ideal_score = 0, 0, 0

                for qid in tqdm(self.queries_run_data):
                    filtered_run_data = {key: value for key, value in self.queries_run_data[qid].items() if key != run_name}
                    eval_score = self.evaluate_query(qid, filtered_run_data, self.queries_run_data[qid][run_name])
                    ideal_score = self.calculate_ideal_ranking(qid, filtered_run_data)
                    normalized_score = eval_score / ideal_score if ideal_score != 0.0 else 0.0

                    total_evaluation_score += eval_score
                    total_normalized_score += normalized_score
                    total_ideal_score += ideal_score

                num_queries = len(self.queries_run_data)
                avg_eval_score = total_evaluation_score / num_queries
                avg_normalized_score = total_normalized_score / num_queries
                avg_ideal_score = total_ideal_score / num_queries

                evaluation_results.append([run_name, avg_eval_score])
                ideal_results.append([run_name, avg_ideal_score])
                normalized_results.append([run_name, avg_normalized_score])
                
                _, total_unique_contributions = self.calculate_unique_contributions()
                unique_contributions.append(total_unique_contributions / num_queries)

            evaluation_df = pd.DataFrame(evaluation_results, columns=["run_name", "avg_evaluation_score"])
            ideal_df = pd.DataFrame(ideal_results, columns=["run_name", "avg_ideal_score"])
            normalized_df = pd.DataFrame(normalized_results, columns=["run_name", "avg_normalized_score"])
            unique_contributions_df = pd.DataFrame(unique_contributions, columns=["avg_num_unique_contributions"])

            evaluation_df.to_csv(os.path.join(self.result_directory, f"{self.dataset_name}_evaluation_scores.csv"), index=False)
            ideal_df.to_csv(os.path.join(self.result_directory, f"{self.dataset_name}_ideal_scores.csv"), index=False)
            normalized_df.to_csv(os.path.join(self.result_directory, f"{self.dataset_name}_normalized_scores.csv"), index=False)
            unique_contributions_df.to_csv(os.path.join(self.result_directory, f"{self.dataset_name}_unique_contributions.csv"), index=False)
        else:
            print("Run files do not contain the same set of queries")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', type=str, required=True, help="Dataset name.")
    parser.add_argument('-run_directory', type=str, required=True, help="Path to the directory where all the run files are located in TREC format.")
    parser.add_argument('-qrels', type=str, required=True, help="Path to the qrels file that consists of relevance judgment documents of queries inside run files in TREC format.")
    parser.add_argument('-doc_gain_version', type=int, required=True, help="Document gain version to use.")
    parser.add_argument('-target_ranker', type=str, required=True, help="Target ranker.")
    parser.add_argument('-competitor_rankers', type=str, nargs='+', default=None, help="List of competitor rankers.")
    parser.add_argument('-result_directory', type=str, required=True, help="Path to the directory where the result files will be saved.")
    
    args = parser.parse_args()

    start_time = time.time()
    evaluator = NormalizedResidualGainCalculator(args.dataset_name, args.run_directory, args.qrels, args.target_ranker, args.competitor_rankers, args.doc_gain_version, args.result_directory)
    evaluator.compare_rankers()
    print(f"--- Total Execution Time: {time.time() - start_time} seconds ---")

if __name__ == "__main__":
    main()
