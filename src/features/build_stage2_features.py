import os
import time
import numpy as np
import pandas as pd

def build_stage2_feature_matrix(
    candidates_path: str,
    output_path: str,
    is_train: bool = True
) -> pd.DataFrame:
    """
    Constructs User, Item, and Interaction features for Stage 1 candidate pairs.
    Appends ground truth binary target labels (1/0) for model training.
    """
    start_time = time.time()
    print("📥 Loading Base Files...")
    
    # 1. Load Candidates
    df_candidates = pd.read_parquet(candidates_path)
    print(f"Loaded {len(df_candidates):,} candidate pairs for feature construction.")

    # 2. Load Raw Metadata
    df_orders = pd.read_csv("data/orders.csv", usecols=[
        'order_id', 'user_id', 'eval_set', 'order_number', 
        'order_dow', 'order_hour_of_day', 'days_since_prior_order'
    ])
    df_prior = pd.read_csv("data/order_products__prior.csv", usecols=[
        'order_id', 'product_id', 'add_to_cart_order', 'reordered'
    ])

    print("⚙️ Pre-processing Historical Orders ('prior' eval set)...")
    
    # Isolate Prior Orders & Merge User Identifiers
    prior_orders = df_orders[df_orders['eval_set'] == 'prior'].copy()
    df_interactions = df_prior.merge(
        prior_orders[['order_id', 'user_id', 'order_number', 'days_since_prior_order']], 
        on='order_id', 
        how='inner'
    )
    
    # Track the maximum order number per user in historical data
    user_max_orders = df_interactions.groupby('user_id')['order_number'].max().to_dict()

    # ==========================================
    # A. USER-LEVEL FEATURES
    # ==========================================
    print("📊 Computing User-Level Features...")
    
    # Total distinct orders placed per user
    user_total_orders = prior_orders.groupby('user_id')['order_number'].max().rename('user_total_orders')
    
    # Average days between orders
    user_avg_days_between = prior_orders.groupby('user_id')['days_since_prior_order'].mean().rename('user_avg_days_between')
    
    # User Basket Size & Reorder Stats
    user_basket_stats = df_interactions.groupby(['user_id', 'order_id']).size().groupby('user_id').agg(
        user_avg_basket_size='mean',
        user_max_basket_size='max',
        user_min_basket_size='min'
    )
    
    user_reorder_stats = df_interactions.groupby('user_id')['reordered'].agg(
        user_overall_reorder_rate='mean',
        user_total_items_bought='count'
    )

    user_features = pd.concat([
        user_total_orders, 
        user_avg_days_between, 
        user_basket_stats, 
        user_reorder_stats
    ], axis=1).reset_index()

    # ==========================================
    # B. ITEM-LEVEL FEATURES
    # ==========================================
    print("📊 Computing Item-Level Features...")
    
    item_features = df_interactions.groupby('product_id').agg(
        item_total_purchases=('order_id', 'count'),
        item_unique_users=('user_id', 'nunique'),
        item_reorder_rate=('reordered', 'mean'),
        item_avg_cart_position=('add_to_cart_order', 'mean')
    ).reset_index()

    # ==========================================
    # C. USER-ITEM INTERACTION FEATURES
    # ==========================================
    print("📊 Computing User-Item Interaction Features...")
    
    # Interaction Counts & Recency
    ui_stats = df_interactions.groupby(['user_id', 'product_id']).agg(
        ui_times_bought=('order_id', 'count'),
        ui_first_order_num=('order_number', 'min'),
        ui_last_order_num=('order_number', 'max'),
        ui_avg_cart_position=('add_to_cart_order', 'mean'),
        ui_reorder_rate=('reordered', 'mean')
    ).reset_index()

    # Map user total orders to compute recency features
    ui_stats['user_max_orders'] = ui_stats['user_id'].map(user_max_orders)
    
    # Orders elapsed since user last bought this specific item
    ui_stats['ui_orders_since_last_buy'] = ui_stats['user_max_orders'] - ui_stats['ui_last_order_num']
    
    # Proportion of user orders since first purchase that included this item
    ui_stats['ui_order_streak_ratio'] = ui_stats['ui_times_bought'] / (
        ui_stats['user_max_orders'] - ui_stats['ui_first_order_num'] + 1
    )
    
    ui_stats.drop(columns=['user_max_orders'], inplace=True)

    # ==========================================
    # D. MERGE ALL FEATURES ONTO CANDIDATES
    # ==========================================
    print("🔗 Merging Features onto Candidate Pool...")
    
    df_dataset = df_candidates.merge(user_features, on='user_id', how='left')
    df_dataset = df_dataset.merge(item_features, on='product_id', how='left')
    df_dataset = df_dataset.merge(ui_stats, on=['user_id', 'product_id'], how='left')

    # Fill NaNs for items never bought by this user before (discovery candidates from Two-Tower)
    df_dataset['ui_times_bought'] = df_dataset['ui_times_bought'].fillna(0)
    df_dataset['ui_reorder_rate'] = df_dataset['ui_reorder_rate'].fillna(0)
    df_dataset['ui_orders_since_last_buy'] = df_dataset['ui_orders_since_last_buy'].fillna(999)
    df_dataset['ui_order_streak_ratio'] = df_dataset['ui_order_streak_ratio'].fillna(0)
    df_dataset['ui_avg_cart_position'] = df_dataset['ui_avg_cart_position'].fillna(99)

    # Add Target Next-Order Context Features
    target_orders = df_orders[df_orders['eval_set'] == ('train' if is_train else 'test')][
        ['user_id', 'days_since_prior_order', 'order_dow', 'order_hour_of_day']
    ].rename(columns={'days_since_prior_order': 'target_days_since_prior_order'})
    
    df_dataset = df_dataset.merge(target_orders, on='user_id', how='left')

    # ==========================================
    # E. ATTACH GROUND TRUTH BINARY TARGET LABELS
    # ==========================================
    if is_train:
        print("🏷️ Attaching Ground Truth Binary Labels (target == 1 or 0)...")
        df_train_labels = pd.read_csv("data/order_products__train.csv", usecols=['order_id', 'product_id'])
        df_train_orders = df_orders[df_orders['eval_set'] == 'train'][['order_id', 'user_id']]
        
        # Actual items purchased in validation order
        df_truth = df_train_labels.merge(df_train_orders, on='order_id', how='inner')[['user_id', 'product_id']]
        df_truth['target'] = 1
        
        # Left merge ground truth to assign 1 to hits, 0 to non-hits
        df_dataset = df_dataset.merge(df_truth, on=['user_id', 'product_id'], how='left')
        df_dataset['target'] = df_dataset['target'].fillna(0).astype(np.int8)
        
        positives = (df_dataset['target'] == 1).sum()
        negatives = (df_dataset['target'] == 0).sum()
        print(f"Target Distribution: {positives:,} Positives (1) | {negatives:,} Negatives (0) | Ratio: 1:{negatives/positives:.2f}")

    # ==========================================
    # F. SAVE & SUMMARY
    # ==========================================
    print(f"💾 Saving Stage 2 Feature Store to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_dataset.to_parquet(output_path, index=False)
    
    print(f"✅ Stage 2 Feature Matrix built in {time.time() - start_time:.2f} seconds!")
    print(f"Dataset Shape: {df_dataset.shape[0]:,} rows × {df_dataset.shape[1]} columns")
    
    return df_dataset

#if __name__ == "__main__":
#    build_stage2_feature_matrix()