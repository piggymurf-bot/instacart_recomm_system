# The Instacart product recommendation system

This project uses the Instacart dataset from Kaggle to train and set up the product recommendation system. The system is divided into two stages: the first stage narrows down the product candidates from about 50,000 items to 50 items for each customer using a two-tower model, and the second stage ingests the product candidates to make a recommendation for each customer, based on buying history. 

---

## 1. Summary & Performance

The performance of the first stage is decided by @Recall50 against all items in the basket. 
  * Evaluating candidate pool across 131,209 validation users (K=50)

  |-------------------------------------------------------|
  |🎯 STAGE 1 RETRIEVAL EVALUATION (K = 50)               |
  |📌 Mean Recall@50:   50.45%                            |
  |📌 Median Recall@50: 50.00%                            |
  |-------------------------------------------------------|
  |📊 Recall Breakdown by True Reorder Basket Size:       |
  |   • Small Baskets (1-5 items):   59.47%               |
  |   • Medium Baskets (6-15 items): 48.36%               |
  |   • Large Baskets (16+ items):   42.64%               |
  |=======================================================|

As for the second stage, it is evaluated using a dynamic F1 score. 



