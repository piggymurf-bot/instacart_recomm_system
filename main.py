import argparse
import os
import sys
from dotenv import load_dotenv


from src.data.download_data import get_instacart_data
from src.models.train_two_towers import two_tower_main
from src.models.stage1_recall import compute_stage1_recall
from src.features.build_stage2_features import build_stage2_feature_matrix
from src.models.train_lightGBM import train_and_optimize_lightgbm
from src.models.dynamics_f1 import run_full_basket_evaluation
from src.models.test_and_gensubmission import verify_and_generate_submission

# 1. Load API keys from environment file
load_dotenv('.kaggle/access_token/token.env')
API_TOKEN = os.getenv('MY_TOKEN')

def main():
  # 2. Parse command-line flags
  parser = argparse.ArgumentParser(
      description='Instacart Recall and Recommendation System'
  )
  parser.add_argument(
      '--download-data',
      action='store_true',
      help='Fetch raw Instacart dataset from KAGGLE APIs',
  )
  parser.add_argument(
      '--stage-one',
      action='store_true',
      help='Set two-tower model and generate candidate Data (train and test)',
  )
  parser.add_argument(
      '--recall50', 
      action='store_true', 
      help='Evaluate @Recall50 on the candidate Data'
  )
  parser.add_argument(
      '--features-building', 
      action='store_true', 
      help='Create features from the candidate Data (train and test)'
  )
  parser.add_argument(
      '--stage-two',
      action='store_true',
      help='Train LightGBM model and evaluate F1 score',
  )
  parser.add_argument(
      '--test',
      action='store_true',
      help='Make predictions on the test set and generate submission',
  )
  parser.add_argument(
      "--all",
      action="store_true",
      help="Run complete pipeline: ",
  )
#  parser.add_argument(
#      "--threshold",
#      type=float,
#      default=0.60,
#      help="Probability threshold for entering long position (default: 0.60)",
# )

  args = parser.parse_args()

  # Handle --all flag shortcut
  if args.all:
    args.download_data = True
    args.stage_one = True
    args.features_building = True
    args.stage_two = True
    args.test = True

  if not any([args.download_data, args.stage_one, args.recall50, args.features_building, args.stage_two, args.test]):
    parser.print_help()
    sys.exit(1)

  # 3. Route execution based on CLI flag
  if args.download_data:
    if not API_TOKEN:
      raise ValueError('API Token missing! Check your token.env file.')
    print('--> Triggering Data Download from Kaggle...')
    get_instacart_data(api_token=API_TOKEN)

  if args.stage_one:
    print('--> Setup Two-Tower model...')
    two_tower_main()
  if args.recall50:
    print('--> Evaluate @Recall50...')
    compute_stage1_recall()

  if args.features_building:
    print('--> Build Features...')
    # Build Training Feature 
    build_stage2_feature_matrix(
        candidates_path = "data/stage1_candidates_top50.parquet",
        output_path = "data/stage2_dataset.parquet",
        is_train=True
    )
    # Build Test Feature 
    build_stage2_feature_matrix(
        candidates_path="data/stage1_test_candidates_top50.parquet",
        output_path="data/stage2_test_dataset.parquet",
        is_train=False
    )

  if args.stage_two:
    print('--> Passing Candidates: Train LightGBM model...')
    best_threshold = train_and_optimize_lightgbm()
    run_full_basket_evaluation(static_best_threshold=best_threshold)

  if args.test:
    print('--> Make prediction and generate submission...')
    verify_and_generate_submission()
   
#  if args.backtest:
#    print(f"Executing Backtesting Engine (Threshold: {args.threshold})...")
    # args.threshold is passed directly here into the engine
    #run_backtest_pipeline(long_threshold=args.threshold)
  
    
if __name__ == '__main__':
  main()