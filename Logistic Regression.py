import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import math

# train_data : qid	pid	queries	passage	relevancy
# validation_data : qid	pid	queries	passage	relevancy

# ---- 1. Load Pretrained Embeddings ----
def load_glove_embeddings(file_path):
    embedding_dict = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict

# ---- 2. Text Preprocessing ----
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().split()

# ---- 3. Compute Average Embeddings ----
def average_embedding(text, embedding_dict, dim):
    words = preprocess(text)
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)   # gives an output shape of (100,)

def prepare_dataset(df, embedding_dict, dim, negative_sampling=False, neg_ratio=4):
    X = []
    y = []
    pid_list = []
    qid_list = []

    if negative_sampling:
        # Group by query ID
        grouped = df.groupby('qid')
        sampled_rows = []

        for qid, group in grouped:
            positives = group[group['relevance'] > 0]
            negatives = group[group['relevance'] == 0]

            # Sample negatives (e.g., 3x the positives or as many as available)
            n_pos = len(positives)
            n_neg_sample = min(len(negatives), n_pos * neg_ratio)

            sampled_negatives = negatives.sample(n=n_neg_sample, random_state=42)
            combined = pd.concat([positives, sampled_negatives])

            combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

            sampled_rows.append(combined)

        df = pd.concat(sampled_rows)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query_vec = average_embedding(row['query'], embedding_dict, dim)
        passage_vec = average_embedding(row['passage'], embedding_dict, dim)
        features = np.concatenate([query_vec, passage_vec])
        X.append(features)
        y.append(row['relevance'])
        pid_list.append(row['pid'])
        qid_list.append(row['qid'])

    return np.array(X), np.array(y), pid_list, qid_list


# ---- 5. Logistic Regression Model ----
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()

# ---- 6. Train Model ----
def train_model(X_train, y_train, learning_rate, epochs):
    model = LogisticRegressionModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model

# ---- 7. Custom Evaluation Metrics ----

# ground thruth is a dictionary for a given query with all the passages(PID values) linked to the query with their relevance value.
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


# ---- 8. Evaluation using Custom Metrics ----
def evaluate_model(model, X_val, y_val, pids, qids, original_df):
    model.eval()
    X_val = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_val).numpy()

    # Group by query
    qrels = defaultdict(dict)
    for idx, row in original_df.iterrows():
        qrels[row['qid']][row['pid']] = row['relevance']

    df_pred = pd.DataFrame({
        'qid': qids,
        'pid': pids,
        'pred_score': predictions
    })

    ap_scores = []
    ndcg_scores = []

    for qid, group in df_pred.groupby('qid'):
        sorted_group = group.sort_values(by='pred_score', ascending=False)
        ranked_pids = sorted_group['pid'].tolist()
        rel_dict = qrels[qid]

        ap = average_precision(ranked_pids, rel_dict)
        ndcg_val = ndcg(ranked_pids, rel_dict)

        ap_scores.append(ap)
        ndcg_scores.append(ndcg_val)

    print(f"Mean Average Precision (MAP): {np.mean(ap_scores):.4f}")
    print(f"Mean NDCG@10: {np.mean(ndcg_scores):.4f}")





def generate_submission_file(model, test_df, embedding_dict, dim, output_filename, algo_name="LR"):
    # Compute features
    qids = test_df['qid'].tolist()
    pids = test_df['pid'].tolist()

    X = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q_vec = average_embedding(row['query'], embedding_dict, dim)
        p_vec = average_embedding(row['passage'], embedding_dict, dim)
        combined = np.concatenate([q_vec, p_vec])
        X.append(combined)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        scores = model(X_tensor).numpy()

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

# ---- 9. Main Script ----
if __name__ == "__main__":
    embedding_dim = 300
    glove_path = "glove.6B/glove.6B.300d.txt"
    embedding_dict = load_glove_embeddings(glove_path)

    train_df = pd.read_csv('dataset2/train_data.tsv', sep='\t', skiprows=1, 
                 names=['qid', 'pid', 'query', 'passage', 'relevance'],
                 dtype={'qid':str, 'pid': str, 'relevance':float})

    val_df = df = pd.read_csv('dataset2/validation_data.tsv', sep='\t', skiprows=1, 
                 names=['qid', 'pid', 'query', 'passage', 'relevance'],
                 dtype={'qid':str, 'pid': str, 'relevance':float})

    print("Preparing training data...")
    # X_train, y_train, _, _ = prepare_dataset(train_df, embedding_dict, embedding_dim)
    X_train, y_train, _, _ = prepare_dataset(train_df, embedding_dict, embedding_dim,
                                         negative_sampling=True, neg_ratio=4)

    print("Preparing validation data...")
    X_val, y_val, val_pids, val_qids = prepare_dataset(val_df, embedding_dict, embedding_dim)

    print("Training model...")
    model = train_model(X_train, y_train, learning_rate=0.05, epochs=30)

    print("Evaluating model with custom IR metrics...")
    evaluate_model(model, X_val, y_val, val_pids, val_qids, val_df)




    test_df = pd.read_csv("dataset2/candidate_passages_top1000.tsv", sep="\t", 
                        names=['qid', 'pid', 'query', 'passage'], dtype=str, skiprows=1)

    generate_submission_file(model, test_df, embedding_dict, dim=embedding_dim, output_filename="LR.txt", algo_name="LR")


# RESULTS
# MAP=0.0200 and NCGD@10=0.0161