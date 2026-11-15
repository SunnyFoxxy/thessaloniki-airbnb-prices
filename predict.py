from __future__ import annotations

import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("model.bin")))
with MODEL_PATH.open("rb") as model_file:
    dv, model, median_data = pickle.load(model_file)
features = dv.get_feature_names_out().tolist()

app = FastAPI(title="ML Zoomcamp 2026 Midterm Project Thessaloniki Airbnb Prices")


class Listing(BaseModel):
    host_is_superhost: Literal['t', 'f']
    host_listings_count: int = Field(ge=0)
    host_has_profile_pic: Literal['t', 'f']
    host_identity_verified: Literal['t', 'f']
    neighbourhood_cleansed: Literal[
        'Ampelokipon – Menemeni',
        'Kalamaria',
        'Kordelio – Evosmos',
        'Neapolis – Sykeon',
        'Pavlou Mela',
        'Pilea – Chortiatis',
        'Thessaloniki',
    ]
    latitude: float
    longitude: float
    room_type: Literal['Entire home/apt', 'Private room', 'Shared room']
    accommodates: int = Field(ge=1)
    bedrooms: int = Field(ge=0)
    beds: float | None = Field(default=None, ge=0)
    minimum_nights: float = Field(ge=1, le=30)
    availability_90: int = Field(ge=0, le=90)
    number_of_reviews: int = Field(ge=0)
    review_scores_rating: float | None = Field(default=None, ge=0, le=5)
    review_scores_location: float | None = Field(default=None, ge=0, le=5)
    calculated_host_listings_count: int = Field(ge=0)
    calculated_host_listings_count_private_rooms: int = Field(ge=0)
    reviews_per_month: float = Field(ge=0)
    host_location_cat: Literal['thessaloniki', 'greece', 'abroad', 'unknown']
    has_reviews: Literal[0, 1]
    license_ok: Literal[0, 1]
    host_has_descr: Literal[0, 1]
    n_amenities: int = Field(ge=0)
    days_since_last_review: float | None = Field(default=None, ge=0)
    days_since_first_review: float | None = Field(default=None, ge=0)
    bathrooms_num: Literal[0, 1]
    bathrooms_shared: Literal[0, 1]
    host_tenure_month: int = Field(ge=0)
    user_tenure_month: int = Field(ge=0)


def get_price(price):

    price = round(float(np.expm1(price)),2)

    return price



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
    
    dmatrix = xgb.DMatrix(X_test, feature_names=features)
    preds = model.predict(dmatrix)
    return {
        "prediction": get_price(preds[0])
    }