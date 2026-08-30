import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import gc

from src.dtypes_list import dtypes

# ==========================================
# 1. DIRECT VRAM DATA SAMPLER (Zero PCIe Overhead)
# ==========================================
class CudaBatchSampler:
    """
    Shuffles and batches data directly on CUDA VRAM.
    Bypasses PyTorch DataLoader and PCIe transfers completely.
    """
    def __init__(self, users_array: np.ndarray, items_array: np.ndarray, batch_size: int = 8192, device: str = "cuda"):
        print("⚡ Pre-loading 32.4M interactions directly into CUDA VRAM...")
        # Move raw arrays to VRAM once
        self.users = torch.from_numpy(users_array).to(device=device, dtype=torch.long)
        self.items = torch.from_numpy(items_array).to(device=device, dtype=torch.long)
        self.batch_size = batch_size
        self.num_samples = len(users_array)
        self.num_batches = self.num_samples // batch_size
        self.device = device

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Permute indices directly inside GPU memory
        indices = torch.randperm(self.num_samples, device=self.device)
        
        for i in range(self.num_batches):
            batch_idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.users[batch_idx], self.items[batch_idx]


# ==========================================
# 2. TWO-TOWER ARCHITECTURE
# ==========================================
class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 64)
        self.user_mlp = nn.Sequential(
            #nn.Linear(64, 128),
            #nn.ReLU(),
            #nn.Linear(128, embed_dim)
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, embed_dim, bias=False)      
        )
        
        self.item_emb = nn.Embedding(num_items, 64)
        self.item_mlp = nn.Sequential(
            #nn.Linear(64, 128),
            #nn.ReLU(),
            #nn.Linear(128, embed_dim)
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, embed_dim, bias=False)
        )

    def forward_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.user_emb(user_ids)
        x = self.user_mlp(x)
        return F.normalize(x, p=2, dim=1)

    def forward_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(item_ids)
        x = self.item_mlp(x)
        return F.normalize(x, p=2, dim=1)


# ==========================================
# 3. FAST TRAINING LOOP (MODERN AMP)
# ==========================================
def train_two_tower(
    model: nn.Module,
    sampler: CudaBatchSampler,
    epochs: int = 5,
    lr: float = 1e-3,
    temperature: float = 0.07,
    device: str = "cuda"
) -> nn.Module:
    model.to(device)
    
    # ⚡ Fused Adam combines parameter updates into 1 CUDA kernel
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n🚀 Starting Fast Training on {device.upper()} ({epochs} Epochs)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for u_batch, i_batch in sampler:
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                u_vecs = model.forward_user(u_batch)
                i_vecs = model.forward_item(i_batch)
                
                # Identify duplicate items in same batch
                item_identity_mask = (i_batch.unsqueeze(0) == i_batch.unsqueeze(1))
                # ⚡ Fused Matrix Multiplication (No memory allocation overhead)
                logits = torch.matmul(u_vecs, i_vecs.T).div_(temperature)

                false_negatives = item_identity_mask & ~torch.eye(len(u_batch), dtype=torch.bool, device=device)

                # ✅ SAFE MASKING: Avoids FP16 underflow/NaN gradient deadlocks
                logits.masked_fill_(false_negatives, -1e4)
                
                labels = torch.arange(len(u_batch), device=device)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(sampler)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
        
    print(f"✅ Training Complete in {time.time() - start_time:.2f} seconds.")
    return model

# ==========================================
# 4. GPU CANDIDATE GENERATION (FAISS)
# ==========================================
def generate_candidates_gpu(
    model: nn.Module,
    user_idx_map: np.ndarray,
    item_idx_map: np.ndarray,
    k: int = 50,
    batch_size: int = 4096,
    device: str = "cuda"
) -> pd.DataFrame:
    model.eval()
    model.to(device)
    
    print("\n⚡ Extracting Embeddings for Vector Search...")
    
    with torch.no_grad():
        item_tensor = torch.arange(len(item_idx_map), dtype=torch.long, device=device)
        item_vectors = model.forward_item(item_tensor).cpu().numpy().astype('float32')
        
        num_users = len(user_idx_map)
        user_vectors_list = []
        for i in range(0, num_users, batch_size):
            u_batch_tensor = torch.arange(i, min(i + batch_size, num_users), dtype=torch.long, device=device)
            u_vecs = model.forward_user(u_batch_tensor).cpu().numpy().astype('float32')
            user_vectors_list.append(u_vecs)
            
        user_vectors = np.vstack(user_vectors_list)

    dim = item_vectors.shape[1]
    print(f"⚡ Running FAISS GPU Search (Top-{k} per user)...")
    cpu_index = faiss.IndexFlatIP(dim)
    
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(item_vectors)
        _, topk_indices = gpu_index.search(user_vectors, k)
        print("✅ GPU Vector Search Complete!")
    except Exception as e:
        print(f"⚠️ FAISS GPU failed ({e}). Falling back to FAISS CPU...")
        cpu_index.add(item_vectors)
        _, topk_indices = cpu_index.search(user_vectors, k)

    print("📦 Constructing candidate table...")
    retrieved_product_ids = item_idx_map[topk_indices]
    
    user_ids_repeated = np.repeat(user_idx_map, k)
    product_ids_flat = retrieved_product_ids.flatten()
    
    return pd.DataFrame({
        'user_id': user_ids_repeated,
        'product_id': product_ids_flat
    })

def generate_hybrid_stage1_candidates(
    df_prior: pd.DataFrame, 
    df_twotower_candidates: pd.DataFrame, 
    top_user_k: int = 35, 
    total_k: int = 50
) -> pd.DataFrame:
    """
    Combines User Purchase History (High Recall) with Two-Tower Predictions (Discovery).
    """
    print("⚡ Extracting Top Reordered Items per User from History...")
    
    # 1. Rank items per user by purchase frequency
    user_history = (
        df_prior.groupby(['user_id', 'product_id'])
        .size()
        .reset_index(name='purchase_count')
    )
    user_history['rank'] = user_history.groupby('user_id')['purchase_count'].rank(method='first', ascending=False)
    
    # Keep top 35 historical items per user
    top_history = user_history[user_history['rank'] <= top_user_k][['user_id', 'product_id']]
    
    print("⚡ Merging History Candidates with Two-Tower Candidates...")
    # 2. Combine history candidates + Two-Tower candidates
    combined = pd.concat([top_history, df_twotower_candidates], ignore_index=True)
    
    # 3. Drop duplicates per user, maintaining top_k limit
    hybrid_candidates = combined.drop_duplicates(subset=['user_id', 'product_id'])
    
    # Restrict to total_k (50) per user
    hybrid_candidates = hybrid_candidates.groupby('user_id').head(total_k).reset_index(drop=True)
    
    return hybrid_candidates
    

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def two_tower_main(
    prior_path: str = "data/order_products__prior.csv",
    orders_path: str = "data/orders.csv",
    output_path: str = "data/stage1_candidates_top50.parquet",
    test_output_path: str = "data/stage1_test_candidates_top50.parquet"
):
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Compute Device: {device.upper()}")
    
    print("\n📥 Loading Instacart Data...")
    df_prior = pd.read_csv(prior_path, usecols=['order_id', 'product_id'], dtype=dtypes)
    df_orders = pd.read_csv(orders_path, usecols=['order_id', 'user_id', 'eval_set'], dtype=dtypes)

    # Merge order metadata into prior items
    df_orders_prior = df_orders[df_orders['eval_set'] == 'prior']
    
    df_interactions = df_prior.merge(df_orders_prior, on='order_id', how='inner')[['user_id', 'product_id']]
    #del df_prior, df_orders
    
    print("🔄 Fast Mapping IDs using pandas Categoricals...")
    # Instant categorical mapping (under 1 second)
    user_cat = df_interactions['user_id'].astype('category')
    item_cat = df_interactions['product_id'].astype('category')
    
    mapped_users = user_cat.cat.codes.values.astype(np.int64)
    mapped_items = item_cat.cat.codes.values.astype(np.int64)
    
    user_idx_map = user_cat.cat.categories.values
    item_idx_map = item_cat.cat.categories.values
    
    num_users = len(user_idx_map)
    num_items = len(item_idx_map)
    print(f"Dataset Stats: {num_users:,} Unique Users | {num_items:,} Unique Items | {len(df_interactions):,} Total Interactions")

    ######### Prepare the dataset for building the test candidate 
    
    df_orders_test = df_orders[df_orders['eval_set'] == 'test']
    
    test_users = df_orders_test['user_id'].unique()
    test_items = df_prior['product_id'].unique()
    
    # Map test user_ids using training categories
    test_user_cats = pd.Categorical(
        test_users, 
        categories=user_idx_map
        )
    mapped_test_users = test_user_cats.codes.astype(np.int64)
    # Map test item_ids using training categories
    test_item_cats = pd.Categorical(
        test_items, 
        categories=item_idx_map
        )
    mapped_test_items = test_item_cats.codes.astype(np.int64)
    
    # Pure GPU Batch Sampler
    sampler = CudaBatchSampler(mapped_users, mapped_items, batch_size=2048, device=device)

    model = TwoTowerModel(num_users=num_users, num_items=num_items, embed_dim=64)
    model = train_two_tower(model, sampler, epochs=5, lr=1e-3, temperature=0.07, device=device)

    # Save model weights
    MODEL_SAVE_PATH = "models/two_tower_model.pt"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
 
    # Clear out python garbage and GPU VRAM
    gc.collect()
    torch.cuda.empty_cache()
 
    print(f"✅ Saved trained Two-Tower model weights to {MODEL_SAVE_PATH}")
    
    df_twotowers_candidates = generate_candidates_gpu(
        model=model,
        user_idx_map=user_idx_map,
        item_idx_map=item_idx_map,
        k=50,
        device=device
    )

    df_candidates = generate_hybrid_stage1_candidates(df_interactions, df_twotowers_candidates, top_user_k = 35, total_k = 50)

    os.makedirs("data", exist_ok=True)
    df_candidates.to_parquet(output_path, index=False)
    
    print(f"\n🎉 SUCCESS! Generated {len(df_candidates):,} candidate pairs.")
    print(f"💾 Saved Stage 1 candidates to: {output_path}")
    
    df_twotowers_test_candidates = generate_candidates_gpu(
        model=model,
        user_idx_map=mapped_test_users,
        item_idx_map=mapped_test_items,
        k=50,
        device=device
    )
    
    df_test_candidates = generate_hybrid_stage1_candidates(df_interactions, df_twotowers_test_candidates, top_user_k = 35, total_k = 50)
    df_test_candidates.to_parquet(test_output_path, index=False)
          
    print(f"\n🎉 SUCCESS! Generated {len(df_test_candidates):,} candidate pairs.")
    print(f"💾 Saved Stage 1 test candidates to: {test_output_path}")
    
  
if __name__ == "__main__":
    two_tower_main()