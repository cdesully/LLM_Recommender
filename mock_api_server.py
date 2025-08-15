# mock_api_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="Mock Telephony Catalog API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example models (Pydantic) for clarity
class Item(BaseModel):
    id: str
    name: str
    price: float
    description: str

class Account(BaseModel):
    account_id: str
    name: str
    balance: float
    description: str

@app.get("/devices", response_model=List[Item])
def devices():
    return [
        Item(
            id="d1",
            name="iPhone 14 Pro",
            price=999.0,
            description=(
                "iPhone 14 Pro — premium performance, ProMotion OLED display, cinematic video, "
                "triple-camera photography system. Ideal for creators and photography enthusiasts. "
                "Long-term OS updates and excellent resale value."
            ),
        ),
        Item(
            id="d2",
            name="Samsung Galaxy S22",
            price=849.0,
            description=(
                "Samsung Galaxy S22 — vibrant AMOLED display, reliable battery, advanced camera "
                "features and a flexible Android experience for power users."
            ),
        ),
        Item(
            id="d3",
            name="iPhone SE (2022)",
            price=429.0,
            description=(
                "iPhone SE — compact and affordable with modern A-series performance. Great for "
                "one-handed use and users wanting iOS on a budget."
            ),
        ),
        Item(
            id="d4",
            name="Google Pixel 7",
            price=599.0,
            description=(
                "Google Pixel 7 — clean Android with best-in-class computational photography, "
                "timely OS updates, and excellent low-light performance."
            ),
        ),
    ]

@app.get("/rateplans", response_model=List[Item])
def rateplans():
    return [
        Item(
            id="p1",
            name="Unlimited Plus",
            price=80.0,
            description="Unlimited talk/text/data with priority data and HD streaming allowances. Best for heavy data users.",
        ),
        Item(
            id="p2",
            name="Family Share 10GB",
            price=60.0,
            description="Shared 10GB pool for up to 4 lines. Good for families who split usage.",
        ),
        Item(
            id="p3",
            name="Basic 5GB",
            price=40.0,
            description="Budget-conscious 5GB plan for light cellular users.",
        ),
        Item(
            id="p4",
            name="Traveler 15GB",
            price=55.0,
            description="Generous data with low-cost international roaming options, good for periodic travel.",
        ),
    ]

@app.get("/addons", response_model=List[Item])
def addons():
    return [
        Item(
            id="a1",
            name="International Calling",
            price=15.0,
            description="Low-cost international calling to many countries — ideal for customers who call overseas frequently.",
        ),
        Item(
            id="a2",
            name="Device Protection",
            price=12.0,
            description="Covers accidental damage, loss and theft with small deductible. Great for travelers.",
        ),
        Item(
            id="a3",
            name="Streaming Package",
            price=8.0,
            description="Bundle of popular streaming services at a discount for heavy media consumers.",
        ),
        Item(
            id="a4",
            name="Hotspot Boost",
            price=10.0,
            description="Extra hotspot data and prioritization for remote work or field usage.",
        ),
    ]

@app.get("/account", response_model=Account)
def account():
    return Account(
        account_id="u123",
        name="John Doe",
        balance=1500.0,
        description="Loyal customer for 5 years, good payment history.",
    )

if __name__ == "__main__":
    uvicorn.run("mock_api_server:app", host="0.0.0.0", port=8000, log_level="info")
