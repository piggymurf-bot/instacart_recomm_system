import os
import shutil
import kagglehub

def get_instacart_data(api_token: str, target_dir: str = "data"):
    os.environ["KAGGLE_KEY"] = api_token
    
    files_to_check = [
        os.path.join(target_dir, "aisles.csv"),
        os.path.join(target_dir, "departments.csv"),
        os.path.join(target_dir, "order_products__prior.csv"),
        os.path.join(target_dir, "order_products__train.csv"),
        os.path.join(target_dir, "orders.csv"),
        os.path.join(target_dir, "products.csv")
    ]

    # Check if ANY file in the list is missing
    if any(not os.path.exists(f) for f in files_to_check):
        print("📥 Dataset missing or incomplete. Downloading from Kaggle...")
        cache_path = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")
        print(f"Dataset downloaded to cache: {cache_path}")
        
        # Ensure target data directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Move/Copy all files from cache to local target_dir
        for file_name in os.listdir(cache_path):
            src_file = os.path.join(cache_path, file_name)
            dest_file = os.path.join(target_dir, file_name)
            
            # Copy file (or use shutil.move if you prefer to relocate instead of duplicate)
            if os.path.isfile(src_file):
                #shutil.copy2(src_file, dest_file) # Copy files from Kaggle .cache to data folder
                shutil.move(src_file, dest_file) # Move files instead
                print(f" Moved: {file_name} ➔ {target_dir}/")
                
        print("✅ All files moved successfully to local data/ folder!")
    else:  
        print("✅ The data already exists in data/. Proceeding to the next step.")
    
    



