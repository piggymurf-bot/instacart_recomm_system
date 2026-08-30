import os
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def evaluate_single_user_recall(user_df: pd.DataFrame, true_items: set) -> tuple:
    """
    Calculates Recall@K and basket metrics for a single user.
    """
    if len(true_items) == 0:
        return None  # Exclude users with 0 reordered items in ground truth

    retrieved_items = set(user_df['product_id'])
    hits = len(retrieved_items.intersection(true_items))
    recall = hits / len(true_items)
    
    return recall, len(true_items)

def evaluate_and_report( 
    df_truth_reorders: pd.DataFrame, 
    df_candidates: pd.DataFrame,
    train_orders: pd.DataFrame, 
    k: int,
    n_jobs: int = -1
):
    """
    Make an evaluation for recall and report
    """ 
    truth_dict = df_truth_reorders.groupby('user_id')['product_id'].apply(set).to_dict()
    
    # Filter candidates to only include evaluation set users
    eval_user_ids = set(train_orders['user_id'])
    df_candidates_eval = df_candidates[df_candidates['user_id'].isin(eval_user_ids)]
    
    num_eval_users = df_candidates_eval['user_id'].nunique()
    print(f"📊 Evaluating candidate pool across {num_eval_users:,} validation users (K={k})...")

    # Group candidate data by user_id
    user_groups = [group for _, group in df_candidates_eval.groupby('user_id')]
    
    # Parallel processing across all CPU cores
    results = Parallel(n_jobs=n_jobs, batch_size=1000)(
        delayed(evaluate_single_user_recall)(
            group, 
            truth_dict.get(user_id, set())
        ) for user_id, group in df_candidates_eval.groupby('user_id')
    )
    
    # Filter out None results (users who had 0 ground truth reorders)
    valid_results = [r for r in results if r is not None]
    recalls, basket_sizes = zip(*valid_results)
    
    mean_recall = float(np.mean(recalls))
    median_recall = float(np.median(recalls))
    
    # 2. Segmented Breakdown by Cart Size
    df_res = pd.DataFrame({'recall': recalls, 'basket_size': basket_sizes})
    
    small_basket_recall = df_res[df_res['basket_size'] <= 5]['recall'].mean()
    med_basket_recall = df_res[(df_res['basket_size'] > 5) & (df_res['basket_size'] <= 15)]['recall'].mean()
    large_basket_recall = df_res[df_res['basket_size'] > 15]['recall'].mean()

    # Print Summary Report
    print("=" * 55)
    print(f"🎯 STAGE 1 RETRIEVAL EVALUATION (K = {k})")
    print("=" * 55)
    print(f"📌 Mean Recall@{k}:   {mean_recall * 100:.2f}%")
    print(f"📌 Median Recall@{k}: {median_recall * 100:.2f}%")
    print("-" * 55)
    print("📊 Recall Breakdown by True Reorder Basket Size:")
    print(f"   • Small Baskets (1-5 items):   {small_basket_recall * 100:.2f}%")
    print(f"   • Medium Baskets (6-15 items): {med_basket_recall * 100:.2f}%")
    print(f"   • Large Baskets (16+ items):   {large_basket_recall * 100:.2f}%")
    print("=" * 55)
    
    return mean_recall

def compute_stage1_recall(
    candidate_path: str = "data/stage1_candidates_top50.parquet",
    k: int = 50,
    n_jobs: int = -1
):
    """
    Evaluates Recall@K of Stage 1 Two-Tower candidate pool against ground truth train set.
    """
    start_time = time.time()
    print(f"📥 Loading Stage 1 Candidates from: {candidate_path}")
    df_candidates = pd.read_parquet(candidate_path)
    
    print("📥 Loading Ground Truth Labels (order_products__train.csv & orders.csv)...")
    df_orders = pd.read_csv('data/orders.csv', usecols=['order_id', 'user_id', 'eval_set'])
    df_train = pd.read_csv('data/order_products__train.csv', usecols=['order_id', 'product_id', 'reordered'])
    
    # 1. Filter ground truth to target evaluation orders (train eval_set)
    train_orders = df_orders[df_orders['eval_set'] == 'train'][['order_id', 'user_id']]
    df_truth = df_train.merge(train_orders, on='order_id', how='inner')

    # Total Basket Ground Truth (All items)
    #truth_all_dict = df_truth.groupby('user_id')['product_id'].apply(set).to_dict()
    
    # Keep only reordered items (target == 1)
    df_truth_reorders = df_truth[df_truth['reordered'] == 1] # Evaluate against Re-order items
    #df_truth_reorders = df_truth # Evaluate against ALL items
     
    # Map user_id to set of true reordered product_ids
    print("\n")
    print("Mean evaluation against All items...")
    mean_recall_all = evaluate_and_report(df_truth, df_candidates, train_orders, k)
    print("\n")
    print("Mean evaluation against RE-ORDER items...")
    mean_recall_reorder = evaluate_and_report(df_truth_reorders, df_candidates, train_orders, k)

    print(f"⏱️ Evaluation completed in {(time.time() - start_time):.2f} seconds.\n")

    return mean_recall_all, mean_recall_reorder

if __name__ == "__main__":
    compute_stage1_recall()