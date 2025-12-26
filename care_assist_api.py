"""
FastAPI application that exposes caregiver assistance endpoints backed by DuckDuckGo.

Endpoints:
    POST /care-info             -> health tips, prevention guidance
    POST /emergency-facilities  -> nearby hospitals / medical facilities
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from duckduckgo_service import (
    DuckDuckGoIntegrationError,
    InfoCard,
    fetch_health_information,
    locate_emergency_facilities,
)

app = FastAPI(
    title="ElderGuard Care Assistance API",
    description=(
        "Provides health information tips and emergency facility lookups using "
        "DuckDuckGo so caregivers can respond faster during incidents."
    ),
    version="0.1.0",
)


class InfoCardResponse(BaseModel):
    title: str
    snippet: str
    url: str
    metadata: Optional[Dict[str, Any]] = None


class HealthInfoRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Subject caregivers need guidance on.")
    audience: str = Field(
        "elderly caregivers",
        description="Audience context (appended to the search query).",
    )
    max_results: int = Field(
        5, ge=1, le=10, description="Number of DuckDuckGo results to surface."
    )

    @validator("topic")
    def normalize_topic(cls, value: str) -> str:
        return value.strip()


class EmergencyLocatorRequest(BaseModel):
    location_query: str = Field(
        ..., min_length=3, description="Human-readable location, e.g. city or neighborhood."
    )
    max_results: int = Field(
        5, ge=1, le=10, description="Number of facilities to surface."
    )

    @validator("location_query")
    def normalize_location(cls, value: str) -> str:
        return value.strip()


def _cards_to_response(cards: List[InfoCard]) -> List[InfoCardResponse]:
    return [
        InfoCardResponse(
            title=card.title,
            snippet=card.snippet,
            url=card.url,
            metadata=card.metadata,
        )
        for card in cards
    ]


@app.get("/", summary="Health assistance service status")
def root() -> Dict[str, Any]:
    return {
        "service": "ElderGuard Care Assistance",
        "endpoints": {
            "health_information": "/care-info",
            "emergency_facilities": "/emergency-facilities",
        },
    }


@app.post("/care-info", response_model=List[InfoCardResponse])
def provide_caregiver_information(payload: HealthInfoRequest) -> List[InfoCardResponse]:
    try:
        cards = fetch_health_information(
            topic=payload.topic,
            max_results=payload.max_results,
            audience=payload.audience,
        )
    except DuckDuckGoIntegrationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _cards_to_response(cards)


@app.post("/emergency-facilities", response_model=List[InfoCardResponse])
def locate_facilities(payload: EmergencyLocatorRequest) -> List[InfoCardResponse]:
    try:
        cards = locate_emergency_facilities(
            location_query=payload.location_query,
            max_results=payload.max_results,
        )
    except DuckDuckGoIntegrationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return _cards_to_response(cards)
