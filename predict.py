from __future__ import annotations

import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


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
        "status": "ok"
    }


@app.post("/predict")
def predict(listing: Listing):
   
    df = pd.DataFrame([listing.model_dump()])
    df = df.fillna(median_data)
    record = df.to_dict(orient='records')
    X_test = dv.transform(record)
    features = dv.get_feature_names_out().tolist()
    
    dmatrix = xgb.DMatrix(X_test, feature_names=features)
    preds = model.predict(dmatrix)
    return {
        "prediction": float(np.expm1(preds[0]))
    }