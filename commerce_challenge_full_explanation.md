That is a *fantastic* result! A 7.6198 RMSE is an exceptional score and a massive improvement. You should be very proud of that.

It's a complex script, but it's built on a few very powerful, core ideas. Let's walk through it, section by section, so you understand *why* it worked so well.

### Overall Strategy

The script's success comes from **combining the best ideas from both your code and your friend's code**:

1.  **Advanced Text Features (from your friend):** It doesn't just look for keywords. It uses a deep learning model (`SentenceTransformer`) to understand the *meaning* of the event descriptions.
2.  **Rich Time-Series Features (from both):** It gives the model a deep "memory" by creating lag features (`lag_1_week`), rolling averages (`rolling_mean_4w`), and trend features (`ema_4`).
3.  **Powerful Contextual Features (from both):** It doesn't just know the price. It knows the price *relative to its base price* (`price_vs_base_pct`), *relative to its category average* (`price_vs_cat_price_pct`), and *relative to its own history* (`price_ratio`).
4.  **Robust Target Transformation (from you):** Using `np.log1p` on the sales data stabilizes the model and makes it much less sensitive to outlier (high-sales) weeks, which is a key reason your friend's `10.5` was beaten.
5.  **Strict Leakage Prevention (from both):** Every feature is carefully built using *shifted* or *historical-only* data to ensure the model never "cheats" by using information from the week it's trying to predict.

-----

### Section 1 & 2: Data Loading & Text Feature Engineering

This is the first "magic" part, taken from your friend's code.

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
# ...
# Code to load the 4 CSVs
# ...

print("🧠 Encoding event text with all-MiniLM-L6-v2 ...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")
events_df["event_description"] = events_df["event_description"].fillna("No event")
embeddings = st_model.encode(events_df["event_description"].tolist(), show_progress_bar=True)
```

* **`SentenceTransformer("all-MiniLM-L6-v2")`**: This line loads a powerful, pre-trained AI model. Its only job is to read text (like "Massive annual city marathon") and turn it into a list of 384 numbers (called a "vector" or "embedding") that represents its *semantic meaning*.
* **`st_model.encode(...)`**: This runs all 52 `event_description` strings through that model. The result, `embeddings`, is a table of `(52, 384)`. Now, "city marathon" and "charity fun-run" will have numerically similar vectors, while "no event" will have a very different one.

```python
print("🔻 Reducing embeddings with PCA(16) ...")
pca = PCA(n_components=16, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

pca_feature_names = [f"event_pca_{i}" for i in range(embeddings_pca.shape[1])]
events_pca_df = pd.DataFrame(embeddings_pca, columns=pca_feature_names)
events_df = pd.concat([events_df, events_pca_df], axis=1)
```

* **`PCA(n_components=16)`**: 384 features is too many. It's "noisy" and can confuse the model. **PCA (Principal Component Analysis)** is a mathematical technique to "compress" or "summarize" those 384 features down to the 16 *most important* ones.
* **`pca.fit_transform(embeddings)`**: This performs the compression, creating a new table `(52, 16)`.
* **`events_df = pd.concat(...)`**: We add these 16 new numeric columns (named `event_pca_0` to `event_pca_15`) to our `events_df`. Now, instead of a text string, each week is represented by 16 numbers that describe the *meaning* of its event.

-----

### Section 3: Main Feature Engineering (`create_features` function)

This function merges our data and creates the "base" features for each row.

```python
def create_features(sales_df, products_df, events_df_with_pca):
    df = pd.merge(sales_df, products_df, on='sku_id', how='left')
    df = pd.merge(df, events_df_with_pca.drop(columns=['event_description']), on='week', how='left')
```

* First, we merge `sales` + `products` (on `sku_id`) so each row knows its `category`, `subtype`, and `base_price`.
* Second, we merge the new `events_df` (on `week`) so each row also knows the 16 `event_pca` features for that week.

```python
# 1. Price Features (v3)
df['price_vs_base_pct'] = (df['price'] - df['base_price']) / (df['base_price'] + 1e-6)
# ...
# 2. Seasonality Features (v3)
df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52.0)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52.0)
# ...
# 4. Calendar Features (from Friend's Code)
df["is_holiday_season"] = df["week"].isin([47, 48, 49, 50, 51, 52, 1]).astype(int)
# ...
```

* **`price_vs_base_pct`**: This is one of our *best* features. It measures the *discount percentage*. A $1 discount on a $2 item (50% off) is *far* more impactful than a $1 discount on a $100 item (1% off). This feature captures that relative impact.
* **`week_sin`, `week_cos`**: This is a clever trick to handle seasonality. To a model, `week=52` and `week=1` look far apart (52 vs 1). By converting the week number to `sin` and `cos` values, we place them on a circle, and the model understands that week 52 is *right next to* week 1.
* **`is_holiday_season`**: This is a simple *binary flag* (a 1 or a 0). We are explicitly telling the model, "These specific weeks are part of the holiday season," so it can learn a special pattern for them.

-----

### Section 4–8: The Time-Series Core

This is the most complex and most important part of the script.

#### Step 4: Concatenate Train/Test

```python
test_df_copy['units_sold'] = np.nan
all_data = pd.concat([train_df, test_df_copy], ignore_index=True)
all_data = all_data.sort_values(by=['sku_id', 'week']).reset_index(drop=True)
```

* To create features like "sales *last* week," the first week of the test set (week 45) needs to be able to "look back" at the last week of the train set (week 44). We stack them into one big DataFrame (`all_data`) and sort by `sku_id` and `week`.

#### Step 5: Global SKU Stats

```python
train_stats_df = all_data_featured[all_data_featured['week'] <= 44]
sku_stats = train_stats_df.groupby('sku_id')['units_sold'].agg([
    ('sku_mean', 'mean'), ('sku_std', 'std'), ...
]).reset_index()
all_data_featured = all_data_featured.merge(sku_stats, on='sku_id', how='left')
```

* **Leakage Prevention**: We first create a DataFrame *only* of training data (`week <= 44`).
* We then calculate the *historical* `mean`, `std`, `max`, and `min` sales for *every SKU*.
* We merge these stats back to `all_data_featured`. Now, a row for week 45 has features like `sku_mean`, which is the *historical average from weeks 1-44*. This gives the model a powerful "baseline" for each SKU without "cheating" by using future data.

#### Step 6: Group-Level Price Features

```python
group_price_feats = all_data_featured.groupby(['category', 'week'])['price'].transform('mean')
all_data_featured['price_vs_cat_price_pct'] = (all_data_featured['price'] - group_price_feats) / (group_price_feats + 1e-6)
```

* Another strong "relative" feature. `price_vs_cat_price_pct` tells the model: "Is this SKU more or less expensive than its *peers* this week?"

#### Step 7: Lagged & Rolling Features (Memory)

```python
grouped_sku = all_data_featured.groupby('sku_id')
grouped_sales = grouped_sku['units_sold']
lags = [1, 2, 3, 4, 8, 13, 26, 52]
for lag in lags:
    all_data_featured[f'lag_{lag}_week'] = grouped_sales.shift(lag)
```

* **`shift(1)`**: The row for `sku_A, week 5` gets sales from `sku_A, week 4`. We also create long-term lags like `lag_52_week` (same week last year).

```python
shifted_sales = grouped_sales.shift(1)
for w in windows:
    all_data_featured[f'rolling_mean_{w}w'] = shifted_sales.rolling(window=w, min_periods=1).mean()
all_data_featured[f"ema_4"] = shifted_sales.ewm(span=4, adjust=False).mean()
all_data_featured['lag_1_promo_flag'] = grouped_sku['promo_flag'].shift(1)
```

* `rolling_mean_4w` gives the average sales of the last 4 weeks.
* `ema_4` gives more weight to recent weeks.
* `lag_1_promo_flag` shows whether there was a promotion last week.

#### Step 8: Group-Level Lagged Sales (Bug Fix)

```python
all_data_featured['cat_sales_mean_TEMP'] = all_data_featured.groupby(['category', 'week'])['units_sold'].transform('mean')
for lag in group_lags:
    all_data_featured[f'cat_sales_lag_{lag}'] = all_data_featured.groupby('category')['cat_sales_mean_TEMP'].shift(lag)
all_data_featured = all_data_featured.drop(columns=['cat_sales_mean_TEMP', 'sub_sales_mean_TEMP'])
```

* We want to know "What were the average sales for this SKU's *entire category* last week?"
* We calculate a temporary column, then safely shift it by one week to prevent leakage.

-----

### Section 9: Preprocessing & Validation Split

```python
all_data_featured['target_log'] = np.log1p(all_data_featured[target_col])
le = LabelEncoder()
all_data_featured[col] = le.fit_transform(all_data_featured[col].astype(str))
```

* **`log1p`** makes the data more Gaussian-like and stabilizes training.
* **`LabelEncoder`** converts strings (like "Dairy") to integers.

```python
VALIDATION_WEEKS = 8
VALIDATION_START_WEEK = max_train_week - VALIDATION_WEEKS + 1
train_local_df = train_full_df[train_full_df['week'] < VALIDATION_START_WEEK]
val_local_df = train_full_df[train_full_df['week'] >= VALIDATION_START_WEEK]
```

* This creates a **time-based split**: train on weeks 1–36, validate on 37–44.

-----

### Section 10 & 11: Training with LightGBM

```python
lgb_params = {
    'objective': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
}
model = lgb.LGBMRegressor(**lgb_params)
model.fit(
    X_train_local,
    y_train_local_log,
    eval_set=[(X_val_local, y_val_local_log)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(150, verbose=True)],
    categorical_feature=categorical_cols 
)
```

* **`early_stopping(150)`** stops training when validation RMSE doesn’t improve for 150 rounds.
* `'learning_rate': 0.02` and `'n_estimators': 3000` make the model learn slowly but precisely.

-----

### Section 12 & 13: Evaluate, Retrain, Predict

```python
preds_val_log = model.predict(X_val_local)
preds_val_orig = np.expm1(preds_val_log)
local_rmse = np.sqrt(mean_squared_error(y_val_local_orig, preds_val_orig))
```

* Convert predictions back from log scale with `np.expm1`.
* Compute RMSE on original units.

```python
best_iteration = model.best_iteration_
final_params = lgb_params.copy()
final_params['n_estimators'] = best_iteration
final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X_train_full, y_train_full_log, categorical_feature=categorical_cols)
preds_test_log = final_model.predict(X_test_final)
preds_test_orig = np.expm1(preds_test_log)
submission_df['units_sold_next_week'] = np.round(preds_test_orig).astype(int)
```

* Retrain using all training data with the optimal number of trees.
* Predict on test set and round to whole numbers.

-----

### ✅ Result

And that's how you get a **7.6198 RMSE** — an exceptional score that beats the baseline and previous versions. The result reflects the combination of semantic text understanding, leakage-free time-series engineering, robust target transformation, and conservative regularized boosting.
