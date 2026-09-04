# The Instacart product recommendation system

This project uses the Instacart dataset from Kaggle to train and set up the product recommendation system. The system is divided into two stages: the first stage narrows down the product candidates from about 50,000 items to 50 items for each customer using a two-tower model, and the second stage ingests the product candidates to make a recommendation for each customer, based on buying history. 

---

## 1. Summary & Performance

In the first stage, the two-tower model reduced the number of possible products to only 50 items. The performance in this stage is evaluated by Recall@50 against all items in the basket. The result for the training set (131,209 customers) is as follows:

  |-------------------------------------------------------|
  |🎯 STAGE 1 RETRIEVAL EVALUATION (K = 50)               |
  |📌 Mean Recall@50:   50.45%                            |
  |📌 Median Recall@50: 50.00%                            |
  |-------------------------------------------------------|
  |📊 Recall Breakdown by True Reorder Basket Size:       |
  |   • Small Baskets (1-5 items):   59.47%               |
  |   • Medium Baskets (6-15 items): 48.36%               |
  |   • Large Baskets (16+ items):   42.64%               |
  |=======================================================|

As for the second stage, the candidate items from the first stage will be used to train a Gradient Boost Decision Tree model to actually give a precise product recommendation. Its performance is evaluated via the dynamic F1 score using Faron's algorithm:

🔍 Optimizing Decision Threshold for Max F1-Score...
Threshold    | Mean Validation F1-Score
------------------------------------------
0.10         | 55.24                    %
0.11         | 55.41                    %
0.12         | 55.55                    %
0.13         | 55.64                    %
0.14         | 55.71                    %
0.15         | 55.70                    %
0.16         | 55.68                    %
0.17         | 55.61                    %
0.18         | 55.52                    %
0.19         | 55.42                    %
0.20         | 55.29                    %
0.21         | 55.15                    %
0.22         | 55.00                    %
0.23         | 54.79                    %
0.24         | 54.55                    %
0.25         | 54.34                    %
0.26         | 54.08                    %
0.27         | 53.84                    %
0.28         | 53.58                    %
0.29         | 53.30                    %
0.30         | 53.01                    %
0.31         | 52.71                    %
0.32         | 52.40                    %
0.33         | 52.12                    %
0.34         | 51.80                    %
==========================================
🎯 OPTIMAL THRESHOLD: 0.14
🏆 BEST VALIDATION F1-SCORE: 55.71%
⏱️ Optimization completed in 186.11 seconds.

🔝 Top 10 Most Important Features:
                      feature   importance
     ui_orders_since_last_buy 4.888528e+06
target_days_since_prior_order 2.425954e+06
                    order_dow 1.733305e+06
        ui_order_streak_ratio 1.562200e+06
              ui_times_bought 1.256616e+06
            order_hour_of_day 4.747572e+05
            item_reorder_rate 2.414873e+05
           ui_first_order_num 1.298308e+05
         ui_avg_cart_position 1.242642e+05
    user_overall_reorder_rate 1.209812e+05
💾 Trained LightGBM Model Saved to: models/stage2_lightgbm.model
📥 Loading Stage 2 Feature Matrix & Raw Target Data...
Validation Users: 41,241 | Total Target Items Across Val Users: 276,978
🔮 Scoring Candidates...

📊 Evaluating Static Threshold against FULL Target Carts...
⚡ Running Faron's Expected F1 Optimizer against FULL Target Carts...

============================================================
🏆 FULL BASKET KAGGLE-EQUIVALENT EVALUATION RESULTS
============================================================
📌 Static Threshold (0.14) Full Cart F1:   55.71%
🚀 Faron's Expected F1 Full Cart F1:       55.74%
📈 Net Absolute Boost from Faron:          +0.02%
============================================================

Lastly, the trained model (both stage one and two) is used on the testing set. The product recommendations for each customer (user ID) are generated and exported to `submission.csv`. 

Max Probability:    1.0000
Mean Probability:   0.0477
Median Probability: 0.0000
% Pairs p >= 0.10:  14.66%

=======================================================
✅ SUBMISSION FILE GENERATED SUCCESSFULLY
=======================================================
📁 File Saved To:          data/submission.csv
📊 Total Rows:             75,000
🛒 Average Predicted Cart: 8.28 items
🚫 Carts Marked 'None':    3,739 (4.99%)
=======================================================

Sample Submission Rows:
 order_id                                             products
  2774568                  39190 47766 21903 18599 43961 17668
   329954                                                 None
  1528013                               21903 38293 8424 27521
  1376945 8309 27959 14947 13176 33572 44632 34658 28465 35948
  1356845             13176 7076 10863 14992 11520 28134 21616
  2161313              196 10441 12427 14715 27839 37710 11266
  1416320       21903 21137 24852 17948 5134 41950 24561 21616
  1735923         17008 31487 35123 15131 34690 12108 2192 196
  1980631              13575 9387 6184 22362 46061 13914 41400
   139655      27845 22935 17794 13176 32655 24964 22963 21903

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
   


