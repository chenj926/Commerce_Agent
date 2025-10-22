# Commerce Data Challenge Overview

Welcome to the commerce domain! This dataset ships everything you need for **three** predictive tasks: next-week demand, personalized recommendations, and coupon redemption. 


## Directory & File Structure

```
├── products.csv                  # 1,500 SKUs (full catalogue)
├── products_sub.csv              # 750 SKUs used in Challenge 2 
├── sales_history_train.csv
├── sales_history_test.csv
├── store_events.csv              # short weekly text memos
├── purchases_train.csv
├── purchases_test.csv
├── customers.csv             # customer attributes used 
├── coupon_offers_train.csv
├── coupon_offers_test.csv
├── session_events.json       # per-customer recent browse actions
└── images/
    └── <sku_id>.png              # only for SKUs in products_sub.csv
```


## Common Reference Tables

### products.csv  (full catalogue, 1,500 rows)

**Columns**

* **sku\_id** — integer product ID
* **category** — top-level category (e.g., “Beverages”, “Dairy”, …)
* **subtype** — finer product type within category
* **base\_price** — reference price (USD)

> `products_sub.csv` has the same columns but only **750** SKUs. Images are provided **only** for these SKUs.


## Challenge 1 — Weekly Store-SKU Demand Forecasting

**Goal**
Predict how many units will be sold next week for each SKU.

**Files**

* `sales_history_train.csv`, `sales_history_test.csv`
* `store_events.csv` 

**Columns (sales\_history\_\*.csv)**

* **sku\_id** — product ID
* **week** — ISO week index (1..52)
* **units\_sold** *(train only)* — units sold this week (target)
* **price** — realized shelf price
* **promo\_flag** — 0/1 whether a promotion was active

**Columns (store\_events.csv)**

* **week** — week index
* **event\_description** — a single sentence about local activity (e.g., *“Local food-truck fair in Riverside on Tuesday.”*)

**Submission**
CSV: `sku_id,week,units_sold_next_week` for all rows in `sales_history_test.csv`.

**Metric**
Root Mean Squared Error (RMSE).


## Challenge 2 — Personalized Product Recommendations

**Goal**
For each customer, recommend top products they are likely to buy.

**Files**

* `purchases_train.csv`, `purchases_test.csv`
* Catalogue subset: `products_sub.csv` (750 SKUs)
* Product images: `images/<sku_id>.png` (for the same 750 SKUs)
* Customer table: `customers.csv`

**Columns (purchases\_\*.csv)**

* **customer\_id** — integer user ID
* **order\_id** — synthetic order identifier
* **month** — purchase month index
* **sku\_id** — purchased SKU ID

**Columns (customers.csv)**

* **customer\_id** — unique integer ID 
* **age\_group** — age bracket: `{18–25, 26–35, 36–50, 51+}`.
* **income\_group** — household income tier: `{low, mid, high}`.
* **signup\_year** — year the customer joined the store program.
* **loyalty\_tier** — membership tier: `{bronze, silver, gold}`.
* **marketing\_opt\_in** — `0/1` flag for consenting to marketing emails and offers.


**Important scope note**
For this challenge, **only consider the 750 SKUs** listed in `products_sub.csv`. The `images/` folder contains one PNG per **those** SKUs only.

**Submission**
Totally—let’s switch Challenge 2 to a **wide** submission. Here’s the drop-in replacement text:

### Submission (wide format)

Provide **one row per customer** found in `purchases_test.csv`, with the **top-10 SKUs in ranked order (best → worst)**:

```
customer_id,sku_id_1,sku_id_2,sku_id_3,sku_id_4,sku_id_5,sku_id_6,sku_id_7,sku_id_8,sku_id_9,sku_id_10
```


**Metric**

We compute **NDCG\@10** using the left-to-right order of `sku_id_1..sku_id_10`.



## Challenge 3 — Coupon Redemption with Session Intent

**Goal**
Predict whether a customer will redeem the coupon for a specific SKU/category.

**Files**

* `coupon_offers_train.csv`, `coupon_offers_test.csv`
* `session_events.json`

**Columns (coupon\_offers\_train.csv)**

* **offer\_id** — unique offer row ID
* **customer\_id** — customer receiving the offer
* **sku\_id** — SKU featured in the offer
* **category** — category of the SKU
* **discount\_pct** — percentage discount offered (5–40)
* **price\_tier** — {low, mid, high}
* **hist\_spend** — historical spend for the customer (USD)
* **email\_open\_rate** — 0–1 open probability
* **avg\_basket\_value** — average order value (USD)
* **target\_redeem** — *(train only)* 0/1 whether the coupon was redeemed

**Columns (coupon\_offers\_test.csv)**
Same as train **without** `target_redeem`.

**Structure (session\_events.json)**

```json
[
  {
    "customer_id": 123,
    "events": [
      { "type": "search",       "q": "oat milk",           "days_ago": 2 },
      { "type": "view_category","category": "Dairy",       "days_ago": 1 },
      { "type": "view_pdp",     "sku_id": 465,             "days_ago": 1 },
      { "type": "wishlist_add", "sku_id": 476,             "days_ago": 0 }
    ]
  }
]
```

* `type` ∈ {`search`, `view_category`, `view_pdp`, `wishlist_add`}
* `days_ago` is recency (0 = today; smaller is more recent).

**Submission**
CSV: `offer_id,target_redeem` for all rows in `coupon_offers_test.csv`.

**Metric**
Macro-F1.



## Notes & Tips

- Only the described columns are provided. Participants must infer any latent variables from provided texts, images or JSON files.
- Ensure submissions strictly adhere to the specified CSV formats.

Good luck and have fun!
