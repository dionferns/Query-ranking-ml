import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score
# import gensim.downloader as api
import os

from tqdm import tqdm

import math

embedding_dim = 300


def load_glove_embeddings(file_path):
    embedding_dict = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict


import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().split()



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
            positives = group_df[group_df['relevance'] > 0]
            negatives = group_df[group_df['relevance'] == 0]

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
        query_vec = average_embedding(row['query'], embedding_dict, dim)
        passage_vec = average_embedding(row['passage'], embedding_dict, dim)
        features = np.concatenate([query_vec, passage_vec])
        X.append(features)
        y.append(row['relevance'])

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

class ReRankingDataset(Dataset):
    def __init__(self, df):
        self.features = df.iloc[:, :-1].values.astype(np.float32)
        self.labels = df.iloc[:, -1].values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# Neural Network
class ReRankingNN(nn.Module):
    def __init__(self, input_dim=600):
        super(ReRankingNN, self).__init__()


        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load data

# def prepare_grouped_dataset():
#     pass

train_df = pd.read_csv('dataset2/train_data.tsv', sep='\t', skiprows=1, 
                names=['qid', 'pid', 'query', 'passage', 'relevance'],
                dtype={'qid':str, 'pid': str, 'relevance':float})

val_df = df = pd.read_csv('dataset2/validation_data.tsv', sep='\t', skiprows=1, 
                names=['qid', 'pid', 'query', 'passage', 'relevance'],
                dtype={'qid':str, 'pid': str, 'relevance':float})

train_df = train_df.sort_values("qid")
val_df = val_df.sort_values("qid")

embedding_dim = 300

glove_path = "glove.6B/glove.6B.300d.txt"
embedding_dict = load_glove_embeddings(glove_path)
X_train, y_train, group_train, train_pids, train_qids = prepare_grouped_dataset(train_df, embedding_dict, dim=embedding_dim, negative_sampling=True, neg_ratio=4)
print("Preparing validation data...")
X_val, y_val, group_val, val_pids, val_qids = prepare_grouped_dataset(val_df, embedding_dict, dim=embedding_dim, negative_sampling=False)


# Split X into query and passage features
query_cols = [f'qf_{i}' for i in range(embedding_dim)]
passage_cols = [f'pf_{i}' for i in range(embedding_dim)]
df_cols = query_cols + passage_cols + ['relevance']

# Combine and convert to DataFrame
train_df = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))), columns=df_cols)
val_df = pd.DataFrame(np.hstack((X_val, y_val.reshape(-1, 1))), columns=df_cols)




# train_df = np.hstack((X_train,y_train.reshape(-1,1)))
# val_df = np.hstack((X_val,y_val.reshape(-1,1)))

print(f"Train df shape {train_df.shape}")
print(f"Val df shape {val_df.shape}")


train_dataset = ReRankingDataset(train_df)
val_dataset = ReRankingDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Train Model
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M1/M2 GPU via MPS backend")
else:
    device = torch.device("cpu")
    print("Using CPU (MPS not available)")

# model = ReRankingNN().to(device)

model = ReRankingNN().to(device)
criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# ---- Evaluate using custom average_precision and ndcg functions ----
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

# Run predictions for all X_val
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    predictions = model(X_tensor).cpu().squeeze().numpy()

# Prepare results grouped by query ID
ap_scores = []
ndcg_scores = []

i = 0  # index pointer
for group_size in group_val:
    # Get slice of predictions, labels, qids, pids for this group
    group_preds = predictions[i:i + group_size]
    group_labels = y_val[i:i + group_size]
    group_pids = val_pids[i:i + group_size]
    group_qids = val_qids[i:i + group_size]

    # Pair pids with their prediction scores
    pid_score_pairs = list(zip(group_pids, group_preds))

    # Rank passages by predicted score
    ranked_pairs = sorted(pid_score_pairs, key=lambda x: x[1], reverse=True)
    ranked_pids = [pid for pid, _ in ranked_pairs]

    # Construct ground truth relevance dict: {pid: relevance}
    ground_truth = dict(zip(group_pids, group_labels))

    ap = average_precision(ranked_pids, ground_truth)
    ndcg_val = ndcg(ranked_pids, ground_truth, k=10)

    ap_scores.append(ap)
    ndcg_scores.append(ndcg_val)

    i += group_size  # move to next query group

# Final custom evaluation results
print(f"\nCustom Metrics Evaluation:")
print(f"Mean Average Precision (MAP): {np.mean(ap_scores):.4f}")
print(f"Mean NDCG@10: {np.mean(ndcg_scores):.4f}")


def average_embedding(text, embedding_dict, dim):
    words = preprocess(text)
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

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
    X = np.array(X)  # Convert list of arrays to a proper NumPy array

    # X_tensor = torch.tensor(X, dtype=torch.float32)
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        scores = model(X_tensor).cpu().numpy().flatten()

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


test_df = pd.read_csv("dataset2/candidate_passages_top1000.tsv", sep="\t", 
                    names=['qid', 'pid', 'query', 'passage'], dtype=str, skiprows=1)

generate_submission_file(model, test_df, embedding_dict, dim=embedding_dim, output_filename="NN.txt", algo_name="NN")


# RESULTS
# Average Precision (MAP): 0.0356 and Mean NDCG@10: 0.0367