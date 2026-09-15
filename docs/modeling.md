# Fare-model design

The primary task predicts fare at pickup. Allowed inputs are passenger count, pickup hour/day, and available pickup coordinate or zone fields. Fare, tip, total amount, distance, and destination information are excluded. Zone IDs are one-hot encoded inside each training fold rather than treated as numeric quantities.

The later 20% of timestamps form the final holdout. Three expanding-window folds tune models on earlier data. Missing features are imputed within folds. XGBRegressor is scored with RMSE; final output includes RMSE, MAE, R², and a median baseline. Logistic regression predicts fixed bands below $10, $10–$25, and at least $25, reporting accuracy and macro precision/recall/F1.

`--stage completed-trip` adds distance and destination data. It answers a different question and must be reported separately. A better completed-trip score is not an improvement in pickup-time prediction.
