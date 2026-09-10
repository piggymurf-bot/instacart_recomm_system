import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from typing import Dict, List, Tuple
from src.models.faron_optimizer import FaronF1Optimizer


def load_dataset(dataset_path: str = "data/stage2_dataset.parquet"):
    print("📥 Loading Stage 2 Feature Matrix...")
    df = pd.read_parquet(dataset_path)

    ignore_cols = {'user_id', 'product_id', 'target', 'order_id'}
    features = [c for c in df.columns if c not in ignore_cols]
    
    unique_users = df['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)
    
    split_idx = int(len(unique_users) * 0.8)
    train_users = set(unique_users[:split_idx])
    
    train_mask = df['user_id'].isin(train_users)
    val_mask = ~train_mask
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, 'target']
    X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, 'target']
    df_val = df.loc[val_mask, ['user_id', 'product_id', 'target']].copy()

    return X_train, y_train, X_val, y_val, df_val, features

def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.Booster:
    print("\n🚀 Training LightGBM Ranker...")
    start_time = time.time()
    random_seed = 42
    
    # Create LightGBM Datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    # Hyperparameters
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
    
    lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Save model to disk for inference script
    lgb_model.save_model("models/stage2_lightgbm.model")
    print(f"✅ LightGBM Training complete in {time.time() - start_time:.2f}s!")
    return lgb_model

def train_catboost(X_train, y_train, X_val, y_val) -> CatBoostClassifier:
    print("\n🚀 Training CatBoost Ranker...")
    start_time = time.time()
    
    # CatBoost native GPU/CPU Pool setup
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.08,
        depth=7,
        eval_metric='Logloss',
        random_seed=42,
        task_type='GPU',  # Fall back to 'CPU' if GPU driver is unavailable
        verbose=100
    )

    cb_model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    cb_model.save_model("models/stage2_catboost.model")
    print(f"✅ CatBoost Training complete in {time.time() - start_time:.2f}s!")
    return cb_model

def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.Booster:
    print("\n🚀 Training XGBoost Ranker...")
    start_time = time.time()
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.08,
        'max_depth': 7,
        'tree_method': 'hist',
        'device': 'cuda',  # Uses GPU acceleration (falls back to CPU if unavailable)
        'seed': 42
    }
    
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=600,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    xgb_model.save_model("models/stage2_xgboost.json")
    print(f"✅ XGBoost Training complete in {time.time() - start_time:.2f}s!")
    return xgb_model

def evaluate_faron_f1(
    df_val: pd.DataFrame, 
    prob_col: str,
    raw_train_path: str = "data/order_products__train.csv",
    orders_path: str = "data/orders.csv"
) -> float:
    # 1. Generate predictions via Faron's Optimizer per user
    user_predictions: Dict[int, List[int]] = {}
    grouped = df_val.groupby('user_id')

    for user_id, group in grouped:
        sorted_group = group.sort_values(by=prob_col, ascending=False)
        items = sorted_group['product_id'].to_numpy()
        probs_u = sorted_group[prob_col].to_numpy()

        best_k = FaronF1Optimizer.get_optimal_k(probs_u)
        user_predictions[user_id] = list(items[:best_k]) if best_k > 0 else []

    # 2. Build Full Basket Ground Truth (Reorders + Net-New Purchases)
    df_train_labels = pd.read_csv(raw_train_path, usecols=['order_id', 'product_id'])
    df_orders = pd.read_csv(orders_path)
    train_orders = df_orders[df_orders['eval_set'] == 'train'][['order_id', 'user_id']]

    val_users = set(df_val['user_id'].unique())
    df_truth_full = df_train_labels.merge(train_orders, on='order_id', how='inner')
    df_truth_full = df_truth_full[df_truth_full['user_id'].isin(val_users)]
    
    truth_grouped = (
        df_truth_full.groupby('user_id')['product_id']
        .apply(set)
        .to_dict()
    )

    # 3. Compute Actual Kaggle F1 Score Across Full Carts
    f1_scores = []
    for user_id in val_users:
        actual_items = truth_grouped.get(user_id, set())
        pred_set = set(user_predictions.get(user_id, []))

        if len(pred_set) == 0 and len(actual_items) == 0:
            f1_scores.append(1.0)
            continue
        if len(pred_set) == 0 or len(actual_items) == 0:
            f1_scores.append(0.0)
            continue

        tp = len(pred_set & actual_items)
        precision = tp / len(pred_set)
        recall = tp / len(actual_items)

        f1_scores.append(0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall))

    return float(np.mean(f1_scores))


def optimize_3way_weights(df_val: pd.DataFrame) -> Tuple[float, float, float, float]:
    print("\n⚡ Searching for Optimal 3-Way Blending Weights...")
    best_f1 = 0.0
    best_weights = (0.33, 0.33, 0.34)

    # Subsample 5,000 validation users specifically for weight optimization to speed up DP evaluation 10x
    sample_users = np.random.choice(df_val['user_id'].unique(), size=min(5000, df_val['user_id'].nunique()), replace=False)
    df_val_sub = df_val[df_val['user_id'].isin(sample_users)].copy()

    # Grid search step = 0.10 (36 combinations instead of 121)
    steps = np.round(np.linspace(0.0, 1.0, 11), 2)
    
    for w_lgb in steps:
        for w_cb in steps:
            if w_lgb + w_cb > 1.0:
                continue
            w_xgb = round(1.0 - w_lgb - w_cb, 2)
            
            df_val_sub['ensemble_prob'] = (
                w_lgb * df_val_sub['lgb_prob'] + 
                w_cb * df_val_sub['cb_prob'] + 
                w_xgb * df_val_sub['xgb_prob']
            )
            
            f1 = evaluate_faron_f1(df_val_sub, 'ensemble_prob')
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w_lgb, w_cb, w_xgb)

    # Run one final evaluation on the FULL validation set using the best weights
    w_lgb, w_cb, w_xgb = best_weights
    df_val['ensemble_prob'] = (
        w_lgb * df_val['lgb_prob'] + 
        w_cb * df_val['cb_prob'] + 
        w_xgb * df_val['xgb_prob']
    )
    final_full_f1 = evaluate_faron_f1(df_val, 'ensemble_prob')

    return w_lgb, w_cb, w_xgb, final_full_f1

def train_ensemble():
    X_train, y_train, X_val, y_val, df_val, features = load_dataset()

    # 1. LightGBM Predictions
    #print("\n📥 Scoring LightGBM...")
    #lgb_model = lgb.Booster(model_file="models/stage2_lightgbm.model")
    
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    df_val['lgb_prob'] = lgb_model.predict(X_val)
    
 
    # 2. CatBoost Predictions
    #print("\n📥 Scoring CatBoost...")
    #cb_model = CatBoostClassifier()
    #cb_model.load_model("models/stage2_catboost.model")

    cb_model = train_catboost(X_train, y_train, X_val, y_val)
    df_val['cb_prob'] = cb_model.predict_proba(X_val)[:, 1]
    
    # 3. Train & Score XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    dval = xgb.DMatrix(X_val)
    df_val['xgb_prob'] = xgb_model.predict(dval)

    # 4. Individual F1 Evaluations
    lgb_f1 = evaluate_faron_f1(df_val, 'lgb_prob')
    cb_f1 = evaluate_faron_f1(df_val, 'cb_prob')
    xgb_f1 = evaluate_faron_f1(df_val, 'xgb_prob')
    
    print("\n📊 Standalone Validation Scores:")
    print(f"  • LightGBM: {lgb_f1 * 100:.2f}%")
    print(f"  • CatBoost: {cb_f1 * 100:.2f}%")
    print(f"  • XGBoost:  {xgb_f1 * 100:.2f}%")

    # 5. Optimize 3-Way Blend
    w_lgb, w_cb, w_xgb, ensemble_f1 = optimize_3way_weights(df_val)
    
    # Save optimal d for processing daily dataset 
    blend_weight = pd.DataFrame([{'w_lgb':w_lgb, 'w_cb':w_cb, 'w_xgb':w_xgb}])
    weight_path = 'data/blend_weight.csv'
    blend_weight.to_csv(weight_path, index=True)

    best_single = max(lgb_f1, cb_f1, xgb_f1)
    print("\n" + "=" * 60)
    print("🏆 3-WAY ENSEMBLE OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Optimal Weights -> LGB: {w_lgb:.2f} | CatBoost: {w_cb:.2f} | XGBoost: {w_xgb:.2f}")
    print(f"Best Single Model F1: {best_single * 100:.2f}%")
    print(f"3-Way Blended F1:     {ensemble_f1 * 100:.2f}%")
    print(f"Net F1 Gain:          +{(ensemble_f1 - best_single) * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    train_ensemble()