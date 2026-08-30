import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict

def compute_user_f1(truth_items: set, pred_items: set) -> float:
    """Calculates F1-score for a single user's predicted vs actual basket."""
    if not truth_items and not pred_items:
        return 1.0
    if not truth_items or not pred_items:
        return 0.0
    
    intersection = len(truth_items.intersection(pred_items))
    if intersection == 0:
        return 0.0
        
    precision = intersection / len(pred_items)
    recall = intersection / len(truth_items)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_threshold_f1(
    df_val: pd.DataFrame, 
    preds: np.ndarray, 
    threshold: float
) -> float:
    """Evaluates mean F1-score across all validation users at a given probability threshold."""
    df_eval = df_val[['user_id', 'product_id', 'target']].copy()
    df_eval['pred_score'] = preds
    
    # Filter items exceeding probability threshold
    df_pred_basket = df_eval[df_eval['pred_score'] >= threshold]

    # Get the list of unique evaluation user_ids in df_eval
    all_users = df_eval['user_id'].unique()
    
    # Code for evaluate F1 score against the repurchase items
    #####################    
    # Group actual vs predicted items per user
    #truth_dict = df_eval[df_eval['target'] == 1].groupby('user_id')['product_id'].apply(set).to_dict()
    #####################
    
    # Code for evaluate F1 score against the full basket items
    #####################
    # Load the ground truth target orders
    df_train_gt = pd.read_csv("data/order_products__train.csv")
    df_orders = pd.read_csv("data/orders.csv")

    # Map order_id to user_id for train orders
    train_orders = df_orders[df_orders['eval_set'] == 'train'][['order_id', 'user_id']]
    df_gt = df_train_gt.merge(train_orders, on='order_id', how='inner')

    # Filter ground truth ONLY for users present in df_eval
    df_gt_eval = df_gt[df_gt['user_id'].isin(all_users)]

    # Full Basket Ground Truth Dictionary
    truth_dict = df_gt_eval.groupby('user_id')['product_id'].apply(set).to_dict()
    ####################
    
    pred_dict = df_pred_basket.groupby('user_id')['product_id'].apply(set).to_dict()
    
    f1_scores = []
    
    for u in all_users:
        u_truth = truth_dict.get(u, set())
        u_pred = pred_dict.get(u, set())
        f1_scores.append(compute_user_f1(u_truth, u_pred))
        
    return float(np.mean(f1_scores))


def train_and_optimize_lightgbm(
    dataset_path: str = "data/stage2_dataset.parquet",
    output_model_path: str = "models/stage2_lightgbm.model",
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[lgb.Booster, float]:
    start_time = time.time()
    print("📥 Loading Stage 2 Feature Matrix...")
    df = pd.read_parquet(dataset_path)

    # 1. Separate Features, Meta Columns, and Target
    ignore_cols = {'user_id', 'product_id', 'target'}
    features = [col for col in df.columns if col not in ignore_cols]
    
    print(f"Features Count: {len(features)} columns")

    # 2. User-Level Train/Validation Split (Prevents User Data Leakage)
    unique_users = df['user_id'].unique()
    np.random.seed(random_seed)
    val_users = np.random.choice(unique_users, size=int(len(unique_users) * val_ratio), replace=False)
    val_mask = df['user_id'].isin(val_users)

    df_train = df[~val_mask]
    df_val = df[val_mask]

    print(f"Train Users: {df_train['user_id'].nunique():,} | Val Users: {df_val['user_id'].nunique():,}")

    X_train, y_train = df_train[features], df_train['target']
    X_val, y_val = df_val[features], df_val['target']

    # 3. Create LightGBM Datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # 4. Hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': random_seed
    }

    # 5. Train LightGBM Model
    print("\n🚀 Training LightGBM Ranker...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # 6. Predict Probabilities on Validation Set
    print("\n🔮 Generating Validation Predictions...")
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    # 7. Grid Search for Optimal Probability Threshold
    print("\n🔍 Optimizing Decision Threshold for Max F1-Score...")
    thresholds = np.arange(0.10, 0.35, 0.01)
    best_threshold = 0.20
    best_f1 = 0.0

    print(f"{'Threshold':<12} | {'Mean Validation F1-Score':<25}")
    print("-" * 42)

    for thresh in thresholds:
        score = evaluate_threshold_f1(df_val, val_preds, thresh)
        print(f"{thresh:<12.2f} | {score * 100:<25.2f}%")
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh

    print("=" * 42)
    print(f"🎯 OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"🏆 BEST VALIDATION F1-SCORE: {best_f1 * 100:.2f}%")
    print(f"⏱️ Optimization completed in {time.time() - start_time:.2f} seconds.")

    # 8. Feature Importance Analysis
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values(by='importance', ascending=False)

    print("\n🔝 Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save model to disk for inference script
    model.save_model(output_model_path)
    print(f"💾 Trained LightGBM Model Saved to: {output_model_path}")

    return best_threshold

if __name__ == "__main__":
    train_and_optimize_lightgbm()