from __future__ import annotations

import hashlib
import os
import pandas as pd
import pickle
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from model import DEFAULTS_PATH, normalize_record

FEATURE_COLUMNS = [
    "host_is_superhost",
    "host_listings_count",
    "host_has_profile_pic",
    "host_identity_verified",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_90",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_location",
    "calculated_host_listings_count",
    "calculated_host_listings_count_private_rooms",
    "reviews_per_month",
    "host_location_cat",
    "has_reviews",
    "license_ok",
    "host_has_descr",
    "n_amenities",
    "days_since_last_review",
    "days_since_first_review",
    "bathrooms_num",
    "bathrooms_shared",
    "host_tenure_month",
    "user_tenure_month",
]

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("model.bin")))
with MODEL_PATH.open("rb") as model_file:
    dv, model, median_data = pickle.load(model_file)

app = FastAPI(title="ML Zoomcamp 2026 Midterm Project Thessaloniki Airbnb Prices")


class Listing(BaseModel):
    host_is_superhost: str
    host_listings_count: int
    host_has_profile_pic: str
    host_identity_verified: str
    neighbourhood_cleansed: str
    latitude: float
    longitude: float
    room_type: str
    accommodates: int
    bedrooms: int
    beds: float | None = None
    minimum_nights: float
    availability_90: int
    number_of_reviews: int
    review_scores_rating: float | None = None
    review_scores_location: float | None = None
    calculated_host_listings_count: int
    calculated_host_listings_count_private_rooms: int
    reviews_per_month: float
    host_location_cat: str
    has_reviews: int
    license_ok: int
    host_has_descr: int
    n_amenities: int
    days_since_last_review: float | None = None
    days_since_first_review: float | None = None
    bathrooms_num: int
    bathrooms_shared: int
    host_tenure_month: int
    user_tenure_month: int



@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_sha256": MODEL_SHA256,
        "feature_defaults_sha256": FEATURE_DEFAULTS_SHA256,
    }


@app.post("/predict")
def predict(lists: list[Listing]):
   
    df = pd.DataFrame([listing.model_dump()])
    df = df.fillna(median_data)
    record = df.to_dict(orient='records')

    dmatrix = xgb.DMatrix(record)
    preds = model.pri=edict(dmatrix)
    return {
        "prediction": preds.tolist()
    }