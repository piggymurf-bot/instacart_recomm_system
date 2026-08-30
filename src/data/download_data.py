import os
import kagglehub

def get_instacart_data(api_token
):
    os.environ["KAGGLE_KEY"] = api_token
    
    files_to_check = [
        "data/aisles.csv",
        "data/departments.csv",
        "data/order_products__prior.csv",
        "data/order_products__train.csv",
        "data/orders.csv",
        "data/products.csv"
    ]

    # Check if ANY file in the list is missing
    if any(not os.path.exists(f) for f in files_to_check):
        # Download latest version
        #os.environ["KAGGLEHUB_CACHE"] = "/data/"
        path = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")
        print("Path to dataset files:", path)
        print("COPY/MOVE data files from Kaggle .Cache folder to the data/")
    else:  
        print("The data already exists. Proceed the next step.")
    
    



