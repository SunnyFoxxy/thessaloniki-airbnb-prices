from __future__ import annotations

import json
import sys

from pathlib import Path
from urllib.request import Request, urlopen

EXAMPLE_PATH = Path(__file__).with_name("example_listing.json")

with EXAMPLE_PATH.open() as f_in:
    listing = json.load(f_in)
    
url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"

request = Request(
    url,
    data=json.dumps(listing).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(request) as response:
    result = json.load(response)

assert set(result) == {"prediction"}
assert result["prediction"] > 0
print(result)