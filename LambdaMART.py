import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, average_precision_score
import xgboost as xgb
from collections import Counter, defaultdict
import os
import math
import torch

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().split()


def load_glove_embeddings(file_path):
    embedding_dict = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict

# ---- 3. Compute Average Embeddings ----

def average_embedding(text, embedding_dict, dim):
    words = preprocess(text)
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)




def prepare_grouped_dataset(df, embedding_dict, dim, negative_sampling=False, neg_ratio=4):
    X, y = [], []
    group = []
    all_pids, all_qids = [], []

    # Optional negative sampling
    if negative_sampling:
        grouped = df.groupby('qid')
        sampled_rows = []

        for qid, group_df in grouped:
            positives = group_df[group_df['relevancy'] > 0]
            negatives = group_df[group_df['relevancy'] == 0]

            if len(positives) == 0:
                continue  # skip queries with no relevant passages

            n_pos = len(positives)
            n_neg_sample = min(len(negatives), n_pos * neg_ratio)
            sampled_negatives = negatives.sample(n=n_neg_sample, random_state=42)


            combined = pd.concat([positives, sampled_negatives])
            combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
            sampled_rows.append(combined)

        df = pd.concat(sampled_rows).sort_values("qid").reset_index(drop=True)

    current_qid = None
    count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query_vec = average_embedding(row['queries'], embedding_dict, dim)
        passage_vec = average_embedding(row['passage'], embedding_dict, dim)
        features = np.concatenate([query_vec, passage_vec])
        X.append(features)
        y.append(row['relevancy'])

        qid = row['qid']
        pid = row['pid']
        all_qids.append(qid)
        all_pids.append(pid)

        if current_qid != qid:
            if current_qid is not None:
                group.append(count)
            count = 1
            current_qid = qid
        else:
            count += 1
    group.append(count)

    print(f"Grouped dataset summary:")
    print(f"  Total rows         : {len(df)}")
    print(f"  Unique query IDs   : {len(set(df['qid']))}")
    print(f"  Groups created     : {len(group)}")
    print(f"  Total samples (X)  : {len(X)}")

    return np.array(X), np.array(y), group, all_pids, all_qids




def average_precision(ranked_list, ground_truth):

    # ranked_list: list of ranked pid values for a given query form by the model.
    # ground_truth: list of unranked pid values for a given query with their relevance value.

    num_relevant = 0
    total_precision = 0.0
    for i, pid in enumerate(ranked_list):
        if ground_truth.get(pid, 0) > 0:
            num_relevant += 1
            total_precision += num_relevant / (i + 1)
    if num_relevant == 0:
        return 0.0
    return total_precision / num_relevant

def dcg(scores):
    return sum([(2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores)])

def ndcg(ranked_list, ground_truth, k=10):
    best_rank = sorted([rel for rel in ground_truth.values()], reverse=True)[:k]
    actual = [ground_truth.get(pid, 0) for pid in ranked_list[:k]]
    idcg = dcg(best_rank)
    if idcg == 0:
        return 0.0
    return dcg(actual) / idcg

# ---- 5. Train LambdaMART Model ----
def train_lambdamart(X_train, y_train, group_train, params, num_rounds):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)
    model = xgb.train(params, dtrain, num_boost_round=num_rounds)
    return model

# ---- 6. Evaluate Model ----
def evaluate_lambdamart(model, X_val, y_val, group_val, pids, qids):
    dval = xgb.DMatrix(X_val)
    preds = model.predict(dval)

    ndcgs = []
    maps = []
    i = 0

    ap_scores = []
    ndcg_scores = []

    for group_size in group_val:
        group_preds = preds[i:i+group_size]
        group_labels = y_val[i:i+group_size]
        group_pids = pids[i:i+group_size]
        group_qids = qids[i:i+group_size]
        ground_truth_relevancy = y_val[i:i+group_size]
        ground_truth_pid = pids[i:i+group_size]

        # Pair each pid with its prediction score
        pid_score_pairs = list(zip(group_pids, group_preds))

        # ground_truth = list(zip(ground_truth_pid, ground_truth_relevancy))
        ground_truth_df = pd.DataFrame({
            'qid': group_qids,
            'pid': ground_truth_pid,
            'relevance': ground_truth_relevancy
        })

        # Sort by prediction score descending
        ranked_pairs = sorted(pid_score_pairs, key=lambda x: x[1], reverse=True)
        # Extract only the ranked PIDs
        ranked_pids = [pid for pid, score in ranked_pairs]
        ground_truth_spec = dict(zip(ground_truth_df['pid'], ground_truth_df['relevance']))

        ap = average_precision(ranked_pids, ground_truth_spec)
        ndcg_val = ndcg(ranked_pids, ground_truth_spec)

        ap_scores.append(ap)
        ndcg_scores.append(ndcg_val)
        i += group_size

    print(f"Mean Average Precision (MAP): {np.mean(ap_scores):.4f}")
    print(f"Mean NDCG@10: {np.mean(ndcg_scores):.4f}")

    return np.mean(ap_scores), np.mean(ndcg_scores)



def generate_submission_file(model, test_df, embedding_dict, dim, output_filename, algo_name="LM"):
    # Compute features
    qids = test_df['qid'].tolist()
    pids = test_df['pid'].tolist()

    X = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q_vec = average_embedding(row['query'], embedding_dict, dim)
        p_vec = average_embedding(row['passage'], embedding_dict, dim)
        combined = np.concatenate([q_vec, p_vec])
        X.append(combined)

    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # model.eval()
    # with torch.no_grad():
    #     scores = model(X_tensor).numpy()

    dtest = xgb.DMatrix(np.array(X))
    scores = model.predict(dtest)

    # Build DataFrame for ranking
    results = pd.DataFrame({
        'qid': qids,
        'pid': pids,
        'score': scores
    })

    # Group by qid and rank
    submission_lines = []
    for qid, group in results.groupby('qid'):
        ranked = group.sort_values(by='score', ascending=False).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.head(100).iterrows(), start=1):
            submission_lines.append(f"{qid} A2 {row['pid']} {rank} {row['score']} {algo_name}")

    # Write to file
    with open(output_filename, 'w') as f:
        f.write("\n".join(submission_lines))
    print(f"Submission file saved: {output_filename}")




# ---- 7. Main Script ----
if __name__ == "__main__":
    embedding_dim = 300
    glove_path = "glove.6B/glove.6B.300d.txt"
    embedding_dict = load_glove_embeddings(glove_path)

    train_df = pd.read_csv("dataset2/train_data.tsv", sep="\t")
    val_df = pd.read_csv("dataset2/validation_data.tsv", sep="\t")

    train_df = train_df.sort_values("qid")
    val_df = val_df.sort_values("qid")

    qid_counts = Counter(val_df['qid'])
    print("Number of qids with only 1 passage:", sum(1 for c in qid_counts.values() if c == 1))
    val_group_sizes = Counter(val_df['qid'])
    num_groups = len(val_group_sizes)
    num_single = sum(1 for qid in val_group_sizes if val_df[val_df['qid'] == qid]['relevancy'].count() == 1)
    num_zero_rel = sum(1 for qid in val_group_sizes if val_df[val_df['qid'] == qid]['relevancy'].sum() == 0)

    print(f"Total groups: {num_groups}")
    print(f"Groups with only 1 passage: {num_single}")
    print(f"Groups with no relevant passages: {num_zero_rel}")




    X_train, y_train, group_train, train_pids, train_qids = prepare_grouped_dataset(train_df, embedding_dict, dim=embedding_dim, negative_sampling=True, neg_ratio=4)
    print("Preparing validation data...")
    X_val, y_val, group_val, val_pids, val_qids = prepare_grouped_dataset(val_df, embedding_dict, dim=embedding_dim, negative_sampling=False)




    print("Training LambdaMART model...")



    print("Starting hyperparameter tuning with random search...")
    param_grid = [
        {"eta": 0.1, "max_depth": 6, "min_child_weight": 0.1, "gamma": 1.0},
        {"eta": 0.05, "max_depth": 4, "min_child_weight": 0.5, "gamma": 0.5},
        {"eta": 0.15, "max_depth": 5, "min_child_weight": 1.0, "gamma": 0.0},
        {"eta": 0.05, "max_depth": 3, "min_child_weight": 0.1, "gamma": 1.0},
        {"eta": 0.2,  "max_depth": 6, "min_child_weight": 0.5, "gamma": 0.2},
    ]


    best_score = -1
    best_model = None
    best_params = None

    for idx, p in enumerate(param_grid):
        print(f"\nTesting configuration {idx + 1}: {p}")
        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg",
            "verbosity": 1,
            **p,
        }
        model = train_lambdamart(X_train, y_train, group_train, params, num_rounds=100)
        ap, ndcg_val = evaluate_lambdamart(model, X_val, y_val, group_val, val_pids, val_qids)

        if ndcg_val > best_score:
            best_score = ndcg_val
            best_model = model
            best_params = p

    print("\nBest hyperparameters found:")
    print(best_params)


    test_df = pd.read_csv("dataset2/candidate_passages_top1000.tsv", sep="\t", 
                        names=['qid', 'pid', 'query', 'passage'], dtype=str, skiprows=1)

    generate_submission_file(best_model, test_df, embedding_dict, dim=embedding_dim, output_filename="LM.txt", algo_name="LM")

# RESULTS
# Mean Average Precision (MAP): 0.0307 and Mean NDCG@10: 0.0317 