import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Set
from src.models.faron_optimizer import FaronF1Optimizer

def compute_user_f1(truth_items: Set[int], pred_items: Set[int]) -> float:
    """Calculates Kaggle F1-score for a single user's predicted vs actual basket."""
    # Special Case: Empty true basket vs empty prediction
    if len(truth_items) == 0 and len(pred_items) == 0:
        return 1.0
    if len(truth_items) == 0 or len(pred_items) == 0:
        return 0.0

    intersection = len(truth_items.intersection(pred_items))
    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_items)
    recall = intersection / len(truth_items)
    return 2.0 * (precision * recall) / (precision + recall)


def run_full_basket_evaluation(
    dataset_path: str = "data/stage2_dataset.parquet",
    raw_train_path: str = "data/order_products__train.csv",
    orders_path: str = "data/orders.csv",
    static_best_threshold: float = 0.2,
    val_ratio: float = 0.2,
    random_seed: int = 42
):
    start_time = time.time()
    print("📥 Loading Stage 2 Feature Matrix & Raw Target Data...")
    
    # 1. Load Feature Store
    df = pd.read_parquet(dataset_path)

    # 2. Build TRUE Full-Basket Ground Truth (Reorders + Net-New First-Time Items)
    df_train_labels = pd.read_csv(raw_train_path, usecols=['order_id', 'product_id'])
    df_orders = pd.read_csv(orders_path)
    train_orders = df_orders[df_orders['eval_set'] == 'train'][['order_id', 'user_id']]

    # Full ground truth mapping (EVERY item bought in the target order)
    df_truth_full = df_train_labels.merge(train_orders, on='order_id', how='inner')
    truth_full_dict: Dict[int, Set[int]] = (
        df_truth_full.groupby('user_id')['product_id']
        .apply(set)
        .to_dict()
    )

    # 3. Validation User Split
    unique_users = df['user_id'].unique()
    np.random.seed(random_seed)
    val_users = np.random.choice(unique_users, size=int(len(unique_users) * val_ratio), replace=False)
    val_mask = df['user_id'].isin(val_users)

    df_val = df[val_mask].copy()

    ignore_cols = {'user_id', 'product_id', 'target'}
    features = [c for c in df.columns if c not in ignore_cols]

    print(f"Validation Users: {len(val_users):,} | Total Target Items Across Val Users: {sum(len(truth_full_dict.get(u, set())) for u in val_users):,}")

    # 4. Load saved LGB model
    model = lgb.Booster(model_file="models/stage2_lightgbm.model")
    
    # 5. Predict Probabilities
    print("🔮 Scoring Candidates...")
    df_val['pred_prob'] = model.predict(df_val[features])

    # =========================================================
    # EVALUATION A: STATIC THRESHOLD (Full Cart Target)
    # =========================================================
    print("\n📊 Evaluating Static Threshold against FULL Target Carts...")
    static_preds = df_val[df_val['pred_prob'] >= static_best_threshold]
    static_pred_dict = static_preds.groupby('user_id')['product_id'].apply(set).to_dict()

    f1_scores_static = []
    for u in val_users:
        u_truth = truth_full_dict.get(u, set())
        u_pred = static_pred_dict.get(u, set())
        f1_scores_static.append(compute_user_f1(u_truth, u_pred))

    mean_f1_static = np.mean(f1_scores_static) * 100

    # =========================================================
    # EVALUATION B: FARON OPTIMIZER (Full Cart Target)
    # =========================================================
    print("⚡ Running Faron's Expected F1 Optimizer against FULL Target Carts...")
    
    f1_scores_faron = []
    val_user_groups = df_val.groupby('user_id')

    for u in val_users:
        u_truth = truth_full_dict.get(u, set())

        if u not in val_user_groups.groups:
            f1_scores_faron.append(compute_user_f1(u_truth, set()))
            continue

        group = val_user_groups.get_group(u)
        sorted_group = group.sort_values(by='pred_prob', ascending=False)
        items = sorted_group['product_id'].to_numpy()
        probs = sorted_group['pred_prob'].to_numpy()

        # Faron dynamic basket length prediction
        best_k = FaronF1Optimizer.get_optimal_k(probs)

        u_pred = set(items[:best_k]) if best_k > 0 else set()
        f1_scores_faron.append(compute_user_f1(u_truth, u_pred))

    mean_f1_faron = np.mean(f1_scores_faron) * 100

    # =========================================================
    # SUMMARY OUTPUT
    # =========================================================
    print("\n" + "=" * 60)
    print("🏆 FULL BASKET KAGGLE-EQUIVALENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"📌 Static Threshold ({static_best_threshold:.2f}) Full Cart F1:   {mean_f1_static:.2f}%")
    print(f"🚀 Faron's Expected F1 Full Cart F1:       {mean_f1_faron:.2f}%")
    print(f"📈 Net Absolute Boost from Faron:          +{mean_f1_faron - mean_f1_static:.2f}%")
    print("=" * 60)
    print(f"⏱️ Full evaluation completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    run_full_basket_evaluation()