# Query-Passage Ranking with Traditional and Neural Models

This project implements a full information retrieval (IR) pipeline for query-passage ranking. It evaluates traditional models (BM25, Logistic Regression), tree-based learning-to-rank (LambdaMART), and neural networks using semantic embeddings and custom relevance metrics. The project was completed as part of the IRDM module and demonstrates model design, evaluation, feature construction, and comparative ranking performance.

---

## Project Structure and Techniques Used

### 1. Retrieval Evaluation and Baseline Benchmarking
- **Model**: BM25 (baseline retrieval model)
- **Evaluation Metrics**:
  - **Mean Average Precision (MAP)**: Measures ranking precision across queries
  - **NDCG@10**: Penalizes relevant documents that appear lower in the ranked list
- **Outcome**:
  - MAP = 0.1183  
  - NDCG@10 = 0.1365

These metrics served as the baseline for comparing more advanced ranking models.

---

### 2. Logistic Regression (Implemented from Scratch)
- **Input Features**: GloVe (300d) embeddings averaged over tokens
  - Query and passage embeddings concatenated into 600d feature vectors
- **Preprocessing**: Lowercasing, punctuation removal, tokenization
- **Class Imbalance Handling**:  
  - Dataset was highly imbalanced (0.1% positive examples)  
  - **Negative Sampling**: 4:1 ratio (non-relevant to relevant)
  - **Shuffling** performed after sampling
- **Model**: Logistic Regression with Sigmoid activation  
- **Loss & Optimizer**: Binary Cross-Entropy with Adam  
- **Hyperparameter Tuning**: Learning rate (best = 0.05)

**Results**:  
MAP = 0.0200, NDCG@10 = 0.0161

---

### 3. LambdaMART (XGBoost)
- **Input Features**: Same 600d query-passage embeddings (GloVe)
- **Training Strategy**: Grouped by query ID; pairwise ranking using `rank:ndcg` objective
- **Hyperparameter Tuning**:
  - Random search over `eta`, `max_depth`, `gamma`, `min_child_weight`
  - Best configuration:
    ```json
    {
      "eta": 0.05,
      "max_depth": 4,
      "min_child_weight": 0.5,
      "gamma": 0.5
    }
    ```

**Results**:  
MAP = 0.0307, NDCG@10 = 0.0317

---

### 4. Feedforward Neural Network (PyTorch)
- **Architecture**: MLP with dense layers  
- **Input**: Same 600d embeddings  
- **Loss**: Mean Squared Error  
- **Optimizer**: Adam  
- **Training**: 30 epochs, batch size = 32  
- **Sampling**: Same 4:1 negative sampling as previous models

**Results**:  
MAP = 0.0356, NDCG@10 = 0.0367

---

## Summary of Results

| Model               | MAP    | NDCG@10 |
|--------------------|--------|---------|
| BM25 (Baseline)     | 0.1183 | 0.1365  |
| Logistic Regression | 0.0200 | 0.0161  |
| LambdaMART          | 0.0307 | 0.0317  |
| Neural Network      | 0.0356 | 0.0367  |

---

## Key Skills Demonstrated
- Information Retrieval and Ranking
- Feature Engineering with GloVe Embeddings
- Model Evaluation: MAP, NDCG@10
- Handling Class Imbalance via Negative Sampling
- Model Implementation:
  - Logistic Regression from scratch
  - XGBoost LambdaMART (learning-to-rank)
  - PyTorch Neural Networks
- Hyperparameter Tuning and Training Stability Analysis

---

## Technologies Used
- Python, NumPy, Pandas
- Scikit-learn, XGBoost
- PyTorch
- GloVe Word Embeddings
- Custom Evaluation Metrics
