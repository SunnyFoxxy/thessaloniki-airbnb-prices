# Thessaloniki Airbnb Prices

Midterm project for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

The model predicts the price per night of an Airbnb listing in Thessaloniki, Greece. The final model is XGBoost, it is served as a web service with FastAPI and packed into a Docker container.

## Problem description

When a host puts a new apartment on Airbnb, the first question is: what price per night to ask? If the price is too high, the apartment stays empty. If it is too low, the host loses money. Most hosts decide by looking at a few similar listings by hand, which is slow and not very accurate.

The price depends on many things together: the location, the size of the apartment, the number of guests, reviews, the host profile, how many amenities there are. It is hard to keep all of them in mind at once.

In this project I train a regression model that predicts the price per night from the listing characteristics. The model can be used as a pricing helper:

- a new host sends the description of the listing to the web service and gets a recommended price;
- an existing host can check if the current price is above or below the market for similar listings.

I chose Thessaloniki because I live here, so I can check if the results make sense.

## Dataset

- Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), Thessaloniki, Central Macedonia, Greece
- Snapshot: 22 September 2026, file `listings.csv.gz`
- The file is committed in this repository: `data/listings.csv.gz` (no download needed)
- Raw data: 4961 listings, 90 columns
- After cleaning: 4600 listings, 30 features
- Target: `price` (price per night in euros). The model is trained on `log1p(price)`

## EDA

The full analysis is in `notebook.ipynb`. Main points:

**Target.** The price has a long tail to the right (skew 4.69). After the log transformation the skew dropped to 0.77, so the models are trained on `log_price`. Prices above 1000 euros were deleted: 2 of them are about 10000 euros and clearly placeholders. The median price is 80 euros.

**Cleaning.**

- deleted columns that are more than 90% empty and columns with only one value;
- deleted listings without price, and 17 listings with `minimum_nights > 30` (long-term rentals, not short stays);
- removed leakage: `estimated_revenue_l365d` (calculated from the price) and all `price_quote_*` columns;
- removed features with correlation > 0.8 between each other (the `availability_*` group, most of the `review_scores_*` group);
- missing values (`beds`, `review_scores_rating`, `review_scores_location`, `days_since_*`) are filled with the median, calculated only on the train part after the split.

**Feature engineering.**

- `host_location_cat`: where the host lives (Thessaloniki / Greece / abroad / unknown);
- flags `has_reviews`, `license_ok`, `host_has_descr` (host wrote a description of 20+ characters);
- `n_amenities`: number of amenities from the amenities list;
- `days_since_first_review`, `days_since_last_review`: counted from the scraping date;
- `host_tenure_month`, `user_tenure_month`: 4 columns with years and months aggregated into 2;
- 2 flags for bathrooms: more than 1 bathroom, shared bathroom.

**Feature importance.**

- No single numeric feature predicts the price strongly. The highest correlation with `log_price` is `bedrooms` (0.50), then `accommodates` (0.44) and `beds` (0.38). The price depends on a combination of features.
- For categorical features I compared the median price of each group with the global median (`risk`). Location matters in both directions: Kalamaria (1.31) and Pilea – Chortiatis (1.21) are more expensive, Ampelokipi – Menemeni (0.75) is cheaper. A private room is about 13% cheaper than the global median. Superhost status, profile picture and identity verification have almost no effect.
- `property_type` was dropped: the large groups add little beyond `room_type`, and the small groups are too small to be reliable.

## Model training

Data split: 60% train / 20% validation / 20% test (`random_state=1`). All parameters were tuned on the validation set. The baseline always predicts the mean `log_price` from the train set.

| Model | Tuned parameters | Validation RMSE | Better than baseline |
|---|---|---|---|
| Baseline (mean) | - | 0.4348 | - |
| Ridge | alpha = 5 | 0.3265 | 25% |
| Random Forest | max_depth = 20, n_estimators = 150, min_samples_leaf = 2, max_features = 0.7 | 0.3017 | 31% |
| **XGBoost** | eta = 0.05, max_depth = 6, min_child_weight = 5, 301 rounds | **0.2926** | **33%** |

RMSE is calculated on `log_price`.

Notes from tuning:

- Ridge: alpha from 5 to 15 gives the same result. Log transformation of skewed features made the result worse, so it is not used.
- Random Forest: full grid search over `max_depth`, `n_estimators`, `min_samples_leaf`, then `max_features` separately.
- XGBoost: tuned step by step: `eta` → `max_depth` → `min_child_weight`. `eta = 0.01` was checked too, but with the tuned parameters it was not better than 0.05.

**Final model: XGBoost.** It was checked once on the test set:

- trained on train only (notebook): test RMSE = 0.3099;
- trained on train + validation (`train.py`): test RMSE = 0.3046.

RMSE 0.30 on the log scale means that the prediction is usually about 1.35 times higher or lower than the real price. Part of the price depends on things that are not in the data (photos, renovation, view), so this is a reasonable result for Airbnb prices.

## Project structure

| File | Description |
|---|---|
| `notebook.ipynb` | data cleaning, feature engineering, EDA, model training and tuning |
| `cleaned_notebook.ipynb` | only the steps needed to train the final model (base for `train.py`) |
| `train.py` | trains the final XGBoost model and saves it to `model.bin` |
| `model.bin` | saved model: DictVectorizer, XGBoost model and medians for missing values |
| `predict.py` | FastAPI web service that loads `model.bin` and returns the price |
| `example_listing.json` | example request (a real listing from the test set, real price 88 euros) |
| `Dockerfile` | Docker image for the web service |
| `pyproject.toml`, `uv.lock` | dependencies (managed with uv) |
| `data/listings.csv.gz` | dataset |

## How to run

### 1. Environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.14.

Install uv (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the dependencies:

```bash
git clone https://github.com/SunnyFoxxy/thessaloniki-airbnb-prices.git
cd thessaloniki-airbnb-prices
uv sync --locked
```

`uv sync` creates the virtual environment in `.venv` and installs the exact versions from `uv.lock`. If Python 3.14 is not installed, uv downloads it.

Activate the environment:

```bash
source .venv/bin/activate
```

On Windows: `.venv\Scripts\activate`. You can also skip the activation and run every command with `uv run`, as in the examples below.

### 2. Notebook

```bash
uv run jupyter notebook
```

Open `notebook.ipynb` and run all cells.

### 3. Train the model

```bash
uv run python train.py
```

Expected output:

```
rmse test: 0.30460481892481467
Model saved to model.bin
```

### 4. Run the web service locally

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

In a second terminal, send the example listing:

```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d @example_listing.json
```

Expected answer:

```json
{"prediction":99.72}
```

The real price of this listing is 88 euros.

Fields `beds`, `review_scores_rating`, `review_scores_location`, `days_since_last_review` and `days_since_first_review` are optional: if they are not in the request, the service fills them with the median from the training data. All other fields are required.

Health check:

```bash
curl http://localhost:9696/health
```

## Docker

Build the image (from the project folder, the dot at the end is needed):

```bash
docker build -t airbnb-price .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 airbnb-price
```

In a second terminal, send the example listing:

```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d @example_listing.json
```

Expected answer: `{"prediction":99.72}`

Stop the container with `Ctrl + C`.

## Cloud deployment

The service is deployed on [Render](https://render.com).

URL: RENDER_URL

Test it:

```bash
curl -X POST RENDER_URL/predict -H "Content-Type: application/json" -d @example_listing.json
```

On the free plan the service sleeps after 15 minutes without requests, so the first request can take about a minute.
