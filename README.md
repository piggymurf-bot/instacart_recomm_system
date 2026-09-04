# Instacart Two-Stage Product Recommendation System

This project uses the Instacart Kaggle dataset to train and set up the product recommendation system. The system is divided into two stages: the first stage narrows down the product candidates from about 50,000 items to 50 items for each customer using a two-tower model, and the second stage ingests the product candidates to make a recommendation for each customer, based on buying history and a dynamic F1 score to match the customer basket's size. 

---

## 1. Summary & Performance

The Instacart dataset consists of about 200,000 customers and 50,000 products, which creates about ten billion possible pairs between customers and products. To better handle the problem, the recommendation engine is split into two distinct stages:

* Stage 1 Candidate Generation (Two-Tower Model): This stage narrows the entire set of products down to 50 candidates per customer based on purchase history. 
* Stage 2 Ranking & Prediction (LightGBM + Faron's Optimizer): This stage ranks the narrowed candidates from the featured buying history and determines the basket size for each customer to maximize an expected F1-score. 

In stage one, Recall@50 against all items in the basket is used to evaluate its performance (over 131,209 customers):

|      Metric            |	            Score       |
|------------------------|-------------------------|
|      Mean Recall@50	   |             50.45%      |
|      Median Recall@50	 |             50.00%      |

  * Small Baskets (1–5 items): 59.47% Recall
  * Medium Baskets (6–15 items): 48.36% Recall
  * Large Baskets (16+ items): 42.64% Recall

As for stage two, the F1-Score (full basket) using Faron's optimization is used to evaluate the performance of the GBDT model in prediction/recommendation (41,241 customers and 276,978 actual reordered products):

|      Stage 2 Decision Strategy                 |     Mean Validation F1-Score       |
|------------------------------------------------|------------------------------------|
| Static Probability Threshold ($p \ge 0.14$)	   |                        55.71%      |
| Dynamic Threshold (Faron's Optimizer)	         |                        55.74%      |

Top 5 features for making predictions 
  1. The recency of the purchase
  2. The repurchase cycle
  3. The day-of-week purchasing pattern
  4. The consistency of repurchasing
  5. The total historical frequency of purchasing

The test predictions on 75,000 customers are given in `submission.csv`
> **Note on Submission Structure:** Each test customer in the Instacart dataset is assigned a single test order_id. Therefore, recommending a basket for an order_id is directly equivalent to generating a product recommendation for that specific customer.


---

## 2. Repository Architecture

```text
instacart_recomm_system/
├── data/                               # Raw datasets, stage candidates, featured dataset, and submission outputs 
├── models/
│   ├── two_tower_model.pt              # Saved stage one two-tower model (PyTorch) 
│   └── stage2_lightgbm.model           # Saved stage two GBDT model (LightGBM)
├── src/
│   ├── __init__.py
│   ├── dtypes_list.py                  # Customized pandas dtypes for each feature
│   ├── data/
│   │   ├── __init__.py
│   │   └── download_data.py            # Downloading Instacart dataset from Kaggle API 
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_stage2_features.py    # Feature engineering according to the stage one candidate  
│   └── models/
│       ├── __init__.py
│       ├── train_two_tower.py           # Setting and training two-tower model and generating stage one candidate 
│       ├── stage1_recall.py             # Evaluating Recall@50 on the stage one candidate
│       ├── train_lightGBM.py            # Training LightGBM model from featured dataset
│       ├── dynamics_f1.py               # Evaluating the dynamic F1 score
│       ├── faron_optimizer.py           # Class object used to perform Faron's optimization
│       └── ttest_and_gensubmission.py   # Employing trained model on the test dataset and generating product recommendation 
├── .Kaggel/access_token/
│   └── TOKEN_FILE                       # Token for Kaggle environment
├── main.py                              # Unified CLI pipeline orchestrator
├── requirements.txt                     # Project dependencies
└── README.md                            # System documentation
```

---
## 3. Machine Learning Pipeline Architecture

                                    [ STAGE 1: RETRIEVAL ]
      User Features ───> User Tower  ──┐
                                       ├──> Inner Product ──> Top-50 Candidates per User
    Product Features ──> Product Tower ┘     (Recall@50)

                                             │
                                             ▼
                                    [ STAGE 2: RANKING ]
    Top-50 Candidates ───> Feature Engineering ───> LightGBM Ranker ───> Predicted Probability P(reorder)
                         (24 UI & Order Features)

                                             │
                                             ▼
                                  [ DECISION & OPTIMIZATION ]
    Predicted Probabilities ───> Faron's F1 Optimizer ───> Dynamic Basket Cut ───> final submission.csv


Candidate Retrieval (Two-Tower Model-PyTorch):
* From the purchase history, it constructs $d$-dimensional vector space shared between customers and products. The more a product is likely to be purchased by a certain customer, the more aligned their vector representations in the shared space. 
* The top 50 product candidates for each user are retrieved once the shared vector space is completely set. 

Feature Engineering Pipeline:
* The raw data is engineered into 24 features, divided into three categories: User Demographics/Behavior, Product Reorder Metrics, and User-Item Interaction Streaks (ui_orders_since_last_buy, ui_order_streak_ratio, ui_avg_cart_position).

Ranking & Threshold Decisioning (GBDT-LightGBM):
* The classifier scores candidate pairs and assigns a calibrated probability threshold $P(\text{reorder})$, then converts probabilities into expected F1 curves per user, dynamically picking the optimal cutoff $k$ items (or predicting None if no candidates cross expected utility thresholds).

---
## 4. Setup & Installation

### Prerequisites

* Python 3.10+
* CUDA-compatible GPU (recommended for Stage 1 training)
* Virtual environment tool (`venv` or `conda`)

```bash
# Clone repository
git clone https://github.com/piggymurf-bot/instacart_recomm_system.git
cd instacart_recomm_system

# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```
> **Note:** Place your kaggle.json API key under ~/.kaggle/ or configure .Kaggle/access_token/ to enable automatic dataset downloading.
---

## 5. Usage & Pipeline Execution

The pipeline is managed via `main.py` CLI interface:

Run Complete End-to-End Pipeline

```bash
python main.py --all
```

Modular Pipeline Commands

```bash
# 1. Download Instacart dataset from Kaggle APIs
python main.py --download-data

# 2. Set two-tower model and generate candidate Data (train and test)
python main.py --stage-one

# 3. Evaluate Recall@50 on the candidate Data
python main.py --recall50

# 4. Create features from the candidate Data (train and test) for feeding into the LightGBM model
python main.py --features-building

# 5. Train the LightGBM model and run Faron F1 optimization
python main.py --stage-two

# 6. Make predictions on the test set and generate submission file
python main.py --test

```

---

## 6. Output Artifacts

Execution yields the following output artifacts:

* Candidates: `data/stage1_candidates_top50.parquet`, `data/stage1_test_candidates_top50.parquet`
* Feature Matrices: `data/stage2_dataset.parquet`, `data/stage2_test_dataset.parquet`
* Trained Weights for Models: `models/two_tower_model.pt`, `models/stage2_lightgbm.model`
* Final Submission: `data/submission.csv`
   
