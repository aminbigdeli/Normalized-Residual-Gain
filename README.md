# Evaluating Relative Retrieval Effectiveness with Normalized Residual Gain
## Introduction

Traditional evaluation metrics, such as NDCG and MRR, have long been the standard for evaluating the effectiveness of information retrieval (IR) systems. These metrics provide absolute measures of effectiveness, allowing for comparisons between the absolute performances of different retrieval methods. However, they do not reveal whether systems with similar performance achieve their results by retrieving the same items or by retrieving different items with similar relevance grades.

To address this limitation, several recent proposals have introduced metrics that measure the relative performance of a retrieval method in the context of results from other methods. In this paper, we introduce a new metric called **Normalized Residual Gain (NRG)**, which extends traditional absolute metrics by adjusting gain values according to the browsing model of the absolute metric, considering the results retrieved by other methods.

Through testing on the MS MARCO dev small and TREC DL 2019 datasets, we find that higher absolute effectiveness does not necessarily correlate with higher NRG scores. In particular, NRG reveals that traditional methods, such as BM25, can still find relevant items missed by modern neural models.

## Repository Structure

This repository contains the implementation of NRG and related evaluation metrics. The code is organized as follows:

- `code/calculate_tasc.py`: Script to calculate Task Subspace Coverage (TaSC).
- `code/calculate_rareness_based_precision.py`: Script to calculate rareness-based precision at K.
- `code/calculate_nrg.py`: Script to calculate normalized residual gain.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- pytrec_eval
- argparse
- tqdm

You can install these dependencies using pip:

```bash
pip install pandas numpy pytrec_eval argparse tqdm
```

### TaSC Calculation
To calculate TaSC metric, we consideir the evaluation metric of all runs to be NDCG@10 and use [calculate_tasc.py](https://github.com/aminbigdeli/Normalized-Residual-Gain/tree/main/code/calculate_tasc.py) script with the appropriate arguments to calculate TaSC scores for TREC runs:
```
python calculate_tasc.py \
     --runs_dir /path/to/runs/dir \
     --qrels_file /path/to/qrels/file \
     --run_group_file /path/to/run_group_names.csv \
     --best_runs_dir /path/to/best_runs/dir \
     --ndcg_cut 10 \
     --result_dir /path/to/results/dir
```
### Rareness-based Precision Calculation
To calculate rareness-based precision at K, run the [calculate_rareness_based_precision.py](https://github.com/aminbigdeli/Normalized-Residual-Gain/tree/main/code/calculate_rareness_based_precision.py) script with the appropriate arguments to calculate rareness-based precision at 10 for TREC runs:
```
python calculate_rareness_based_precision.py \
     --run_dir /path/to/runs/dir \
     --qrels_file /path/to/qrels/file \
     --result_dir /path/to/results/dir \
     --alpha 1.0 \
     --k 10
```
### Normalized Residual Gain Calculation
To calculate normalized residual gain, run the [normalized_residual_gain.py](https://github.com/aminbigdeli/Normalized-Residual-Gain/tree/main/code/calculate_nrg.py) script with the appropriate arguments on MS MARCO dev small and TREC DL 2019 runs:
```
python calculate_nrg.py \
     --dataset_name DATASET_NAME \ 
     --runs_dir /path/to/runs/dir \
     --qrels /path/to/qrels/file \
     --doc_gain_version 2 \
     --target_ranker TARGET_RANKER \
     --competitor_rankers COMPETITOR1 COMPETITOR2 \
     --result_directory /path/to/results/dir
```

### Example
The figure below compares NDCG@10 vs. NRG for all runs submitted to the passage-retrieval task of the TREC 2019 Deep Learning Track. For each run, the prior set for computing NRG consists of the best run submitted by each group. Generally, higher NDCG values correlate with higher NRG values. However, non-neural BM25-based runs, particularly those from the "BASELINE" group, deviate from this trend. Despite their lower NDCG@10 values, these runs exhibit higher NRG values.

<p align="center">
  <img src="https://github.com/aminbigdeli/Normalized-Residual-Gain/blob/main/trecdl2019_NRG_VS_NDCG.png">
</p>
