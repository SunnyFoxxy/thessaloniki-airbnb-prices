from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen


url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"
listing = {
  "host_is_superhost": "f",
  "host_listings_count": 13,
  "host_has_profile_pic": "t",
  "host_identity_verified": "t",
  "neighbourhood_cleansed": "Thessaloniki",
  "latitude": 40.62884,
  "longitude": 22.9505,
  "room_type": "Entire home/apt",
  "accommodates": 3,
  "bedrooms": 1,
  "beds": 1.0,
  "minimum_nights": 1.0,
  "availability_90": 90,
  "number_of_reviews": 8,
  "review_scores_rating": 5.0,
  "review_scores_location": 5.0,
  "calculated_host_listings_count": 13,
  "calculated_host_listings_count_private_rooms": 0,
  "reviews_per_month": 0.13,
  "host_location_cat": "thessaloniki",
  "has_reviews": 1,
  "license_ok": 1,
  "host_has_descr": 0,
  "n_amenities": 13,
  "days_since_last_review": 137.0,
  "days_since_first_review": 1895.0,
  "bathrooms_num": 0,
  "bathrooms_shared": 0,
  "host_tenure_month": 112,
  "user_tenure_month": 125
}

request = Request(
    url,
    data=json.dumps(listing).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(request) as response:
    result = json.load(response)

assert set(result) == {"prediction"}

print(result)