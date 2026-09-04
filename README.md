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
├── data/                               # Store all data files (raw data, stage 1 candidate, stage 2 featured data, and submission file 
├── models/
│   ├── two_tower_model.pt              # Stage one saved two-tower model 
│   └── stage2_lightgbm.model           # Stage two saved GBDT model (LightGBM here)
├── src/
│   ├── __init__.py
│   ├── dtypes_list.py                  # Customized data types for each feature
│   ├── data/
│   │   ├── __init__.py
│   │   └── download_data.py            # Fetching Instacart dataset from Kaggle
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_stage2_features.py    # Featuring the dataset according to the stage one candidate before feeding into stage two 
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
## 3. Machine Learning Pipeline

---
## 4. Setup & Installation

### Prerequisites

* Python 3.10+
* Virtual environment tool (`venv` or `conda`)

```bash
# Clone repository
git clone [https://github.com/your-username/btc_stacking_system.git](https://github.com/your-username/btc_stacking_system.git)
cd btc_stacking_system

# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

---

## 5. Usage & Pipeline Execution

The pipeline is managed via `main.py` CLI interface:

Run Complete End-to-End Pipeline

```bash
python main.py --all
```

Modular Pipeline Commands

```bash
# 1. Fetch raw Instacart dataset from KAGGLE APIs
python main.py --download-data

# 2. Set two-tower model and generate candidate Data (train and test)
python main.py --stage-one

# 3. Create features from the candidate Data (train and test) for feeding into the LightGBM model
python main.py --features-building

# 4. Train LightGBM model and evaluate F1 score
python main.py --stage-two

# 5. Make predictions on the test set and generate submission
python main.py --test

# 6. Evaluate Recall@50 on the candidate Data
python main.py --recall50

```

---

## 6. Output Artifacts

The pipeline execution gives out the following output artifacts:

* Stage one candidate data: `data/stage1_candidates_top50.parquet` and  `data/stage1_test_candidates_top50.parquet`
* Stage two featured data: `data/stage2_dataset.parquet` and  `data/stage2_test_dataset.parquet`.
* Trained Model Binary: `models/two_tower_model.pt` and `models/stage2_lightgbm.model`
* Prediction on test dataset: `data/submission.csv` 
   


