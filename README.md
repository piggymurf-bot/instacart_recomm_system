# The Instacart product recommendation system

This project uses the Instacart dataset from Kaggle to train and set up the product recommendation system. The system is divided into two stages: the first stage narrows down the product candidates from about 50,000 items to 50 items for each customer using a two-tower model, and the second stage ingests the product candidates to make a recommendation for each customer, based on buying history. 

---

## 1. Summary & Performance

The performance of the first stage is decided by Recall@50 against all items in the basket. 
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


🔍 Updated Probability Distribution Check:
Max Probability:    1.0000
Mean Probability:   0.0477
Median Probability: 0.0000
% Pairs p >= 0.10:  14.66%

⚡ Optimizing Dynamic Carts via Faron's Algorithm...
📝 Formatting Submission File...

=======================================================
✅ SUBMISSION FILE GENERATED SUCCESSFULLY
=======================================================
📁 File Saved To:          data/submission.csv
📊 Total Rows:             75,000
🛒 Average Predicted Cart: 8.28 items
🚫 Carts Marked 'None':    3,739 (4.99%)
=======================================================

Sample Submission Rows:
 order_id                                             products
  2774568                  39190 47766 21903 18599 43961 17668
   329954                                                 None
  1528013                               21903 38293 8424 27521
  1376945 8309 27959 14947 13176 33572 44632 34658 28465 35948
  1356845             13176 7076 10863 14992 11520 28134 21616
  2161313              196 10441 12427 14715 27839 37710 11266
  1416320       21903 21137 24852 17948 5134 41950 24561 21616
  1735923         17008 31487 35123 15131 34690 12108 2192 196
  1980631              13575 9387 6184 22362 46061 13914 41400
   139655      27845 22935 17794 13176 32655 24964 22963 21903


