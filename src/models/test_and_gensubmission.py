import time
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from typing import Dict, List
from src.models.faron_optimizer import FaronF1Optimizer

# ---------------------------------------------------------
# 2. RUN SUBMISSION WITH PROPER TEST FEATURES
# ---------------------------------------------------------
def verify_and_generate_submission(
    test_dataset_path: str = "data/stage2_test_dataset.parquet",
    orders_path: str = "data/orders.csv",
    output_csv: str = "data/submission.csv"
):
    if not os.path.exists(test_dataset_path):
       raise FileNotFoundError(
           f'Missing raw data file: {test_dataset_path}. '
       )

    print("\n📥 Loading Test Feature Matrix...")
    df_test = pd.read_parquet(test_dataset_path)
    ignore_cols = {'user_id', 'product_id', 'order_id'}
    features = [c for c in df_test.columns if c not in ignore_cols]

    # Single LightGBM Classifier for stage 2
    ###########
    # Load saved LGB model
    #model = lgb.Booster(model_file="models/stage2_lightgbm.model")
    # Predict
    #print("🔮 Scoring Test Candidates...")
    #df_test['pred_prob'] = model.predict(df_test[features])
    ###########

    # Ensemble Classifier for stage 2
    ###########
    lgb_model = lgb.Booster(model_file="models/stage2_lightgbm.model")
    cb_model = CatBoostClassifier()
    cb_model.load_model("models/stage2_catboost.model")
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/stage2_xgboost.json")

    # Predict test probabilities
    lgb_probs = lgb_model.predict(df_test[features])
    cb_probs = cb_model.predict_proba(df_test[features])[:, 1]
    xgb_probs = xgb_model.predict(xgb.DMatrix(df_test[features]))

    # Blend using optimal weights 
    blend_weight = pd.read_csv("data/blend_weight.csv") #Getting optimal weight from training step
    w_lgb = blend_weight['w_lgb'][0]
    w_cb = blend_weight['w_cb'][0]
    w_xgb = blend_weight['w_xgb'][0]
    df_test['pred_prob'] = (w_lgb * lgb_probs) + (w_cb * cb_probs) + (w_xgb * xgb_probs)
    ###########
   
    # Check distribution
    probs = df_test['pred_prob'].to_numpy()
    print("\n🔍 Updated Probability Distribution Check:")
    print(f"Max Probability:    {probs.max():.4f}")
    print(f"Mean Probability:   {probs.mean():.4f}")
    print(f"Median Probability: {np.median(probs):.4f}")
    print(f"% Pairs p >= 0.10:  {(probs >= 0.10).mean() * 100:.2f}%\n")

    # Import Faron Optimizer
    #from faron_optimizer import FaronF1Optimizer

    print("⚡ Optimizing Dynamic Carts via Faron's Algorithm...")
    user_predictions: Dict[int, List[str]] = {}
    grouped = df_test.groupby('user_id')

    for user_id, group in grouped:
        sorted_group = group.sort_values(by='pred_prob', ascending=False)
        items = sorted_group['product_id'].to_numpy()
        probs_u = sorted_group['pred_prob'].to_numpy()

        best_k = FaronF1Optimizer.get_optimal_k(probs_u)

        if best_k > 0:
            user_predictions[user_id] = [str(pid) for pid in items[:best_k]]
        else:
            user_predictions[user_id] = []

    print("📝 Formatting Submission File...")
    df_orders = pd.read_csv(orders_path)
    test_orders = df_orders[df_orders['eval_set'] == 'test'][['order_id', 'user_id']]

    submission_rows = []
    for _, row in test_orders.iterrows():
        u_id = row['user_id']
        o_id = row['order_id']
        preds = user_predictions.get(u_id, [])
        prod_str = " ".join(preds) if len(preds) > 0 else "None"
        submission_rows.append({'order_id': o_id, 'products': prod_str})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(output_csv, index=False)

    empty_carts = (df_sub['products'] == 'None').sum()
    non_empty_mask = df_sub['products'] != 'None'
    avg_cart_len = (
        df_sub[non_empty_mask]['products']
        .apply(lambda x: len(str(x).split()))
        .astype('int64')
        .mean()
        if non_empty_mask.any() else 0.0
    )

    print("\n" + "=" * 55)
    print("✅ SUBMISSION FILE GENERATED SUCCESSFULLY")
    print("=" * 55)
    print(f"📁 File Saved To:          {output_csv}")
    print(f"📊 Total Rows:             {len(df_sub):,}")
    print(f"🛒 Average Predicted Cart: {avg_cart_len:.2f} items")
    print(f"🚫 Carts Marked 'None':    {empty_carts:,} ({empty_carts / len(df_sub) * 100:.2f}%)")
    print("=" * 55)
    print("\nSample Submission Rows:")
    print(df_sub.head(10).to_string(index=False))


if __name__ == "__main__":
    verify_and_generate_submission()