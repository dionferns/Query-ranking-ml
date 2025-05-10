import pandas as pd
from collections import defaultdict
import math

# Load validation data
df = pd.read_csv('dataset2/validation_data.tsv', sep='\t', skiprows=1, 
                 names=['qid', 'pid', 'query', 'passage', 'relevance'],
                 dtype={'qid':str, 'pid': str, 'relevance':float})
# sep='\t' will read the first line of the file as the headers and so will ignore it.
# Group relevance info by query.

#Code below lets you look up relevance of a passage to a query by typing qrels[#qid][#pid]
qrels = defaultdict(dict)
for _, row in df.iterrows():
    qrels[row['qid']][row['pid']] = row['relevance']


#computes the precision at each rank where a relevent item is found.
def average_precision(ranked_list, ground_truth):
    num_relevant = 0
    total_precision = 0.0
    for i, pid in enumerate(ranked_list):
        if ground_truth.get(pid, 0) > 0:
            num_relevant += 1
            total_precision += num_relevant / (i + 1)
    if num_relevant == 0:
        return 0.0
    return total_precision / num_relevant

# NDCG assesed how well a ranked list of documents places the relevent documents near the top 
def dcg(scores):
    return sum([(2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores)])

def ndcg(ranked_list, ground_truth, k=10):
    best_rank = sorted([rel for rel in ground_truth.values()], reverse=True)[:k]
    actual = [ground_truth.get(pid, 0) for pid in ranked_list[:k]]
    idcg = dcg(best_rank)
    if idcg == 0:
        return 0.0
    return dcg(actual) / idcg

# ----- BM25 Retrieval -----
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

query_groups = df.groupby('qid')

ap_scores = []
ndcg_scores = []

for qid, group in query_groups:
    query = group['query'].iloc[0]
    passages = group['passage'].tolist()

    # Tokenize only the passages for this query
    tokenized_passages = [word_tokenize(p.lower()) for p in passages]
    bm25 = BM25Okapi(tokenized_passages)
    tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(tokens)  # scores will match group size 
    # scores will be a list of 1000 scores, one for each passage for the given query

    group = group.copy()
    group['bm25_score'] = scores
    group = group.sort_values('bm25_score', ascending=False)

    ranked_pids = group['pid'].tolist()
    rel_dict = qrels[qid]

    ap = average_precision(ranked_pids, rel_dict)
    ndcg_val = ndcg(ranked_pids, rel_dict)

    ap_scores.append(ap)
    ndcg_scores.append(ndcg_val)
print(f"Mean Average Precision (MAP): {sum(ap_scores) / len(ap_scores):.4f}")
print(f"Mean NDCG: {sum(ndcg_scores) / len(ndcg_scores):.4f}")

# RESULTS
# Mean Average Precision (MAP): 0.1183
# Mean NDCG: 0.1365
