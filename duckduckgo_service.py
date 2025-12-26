"""
DuckDuckGo-powered caregiver assistance utilities.

This module now relies on DuckDuckGo's Instant Answer API instead of the
deprecated `duckduckgo_search` client. The API returns structured JSON that we
normalize into `InfoCard` objects for the rest of the application.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

IA_ENDPOINT = "https://api.duckduckgo.com/"
IA_DEFAULT_PARAMS = {
    "format": "json",
    "no_html": "1",
    "no_redirect": "1",
    "skip_disambig": "1",
}
USER_AGENT = "ElderGuard/1.0 (+https://github.com/petitmj/epital-elderguard)"


class DuckDuckGoIntegrationError(RuntimeError):
    """Raised when the DuckDuckGo search backend fails."""


@dataclass
class InfoCard:
    """Normalized structure returned by DuckDuckGo helper functions."""

    title: str
    snippet: str
    url: str
    metadata: Optional[Dict[str, Any]] = None


def _call_instant_answer(query: str) -> Dict[str, Any]:
    """Call the DuckDuckGo Instant Answer API and return the JSON payload."""
    params = {"q": query, **IA_DEFAULT_PARAMS}
    try:
        response = requests.get(
            IA_ENDPOINT, params=params, headers={"User-Agent": USER_AGENT}, timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network issues
        raise DuckDuckGoIntegrationError(
            f"DuckDuckGo Instant Answer request failed for query='{query}'."
        ) from exc

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - unexpected API response
        raise DuckDuckGoIntegrationError("DuckDuckGo returned invalid JSON.") from exc


def _flatten_related_topics(related_topics: Iterable[Any]) -> List[Dict[str, Any]]:
    """Flatten the 'RelatedTopics' structure returned by the Instant Answer API."""
    flattened: List[Dict[str, Any]] = []

    for topic in related_topics:
        if "Topics" in topic:
            flattened.extend(topic["Topics"])
        else:
            flattened.append(topic)

    return flattened


def _cards_from_instant_answer(
    data: Dict[str, Any], fallback_title: str, limit: int
) -> List[InfoCard]:
    """Convert Instant Answer payload into a list of InfoCards."""
    cards: List[InfoCard] = []

    abstract_text = data.get("AbstractText")
    abstract_url = data.get("AbstractURL") or data.get("AbstractSource") or ""
    if abstract_text:
        cards.append(
            InfoCard(
                title=data.get("Heading") or fallback_title,
                snippet=abstract_text,
                url=abstract_url,
                metadata={
                    "source": data.get("AbstractSource"),
                    "image": data.get("Image"),
                },
            )
        )

    related_topics = data.get("RelatedTopics") or []
    for topic in _flatten_related_topics(related_topics):
        if len(cards) >= limit:
            break
        cards.append(
            InfoCard(
                title=topic.get("Text") or fallback_title,
                snippet=topic.get("Text") or "",
                url=topic.get("FirstURL") or "",
            )
        )

    return cards[:limit]


def fetch_health_information(
    topic: str, max_results: int = 5, audience: str = "elderly caregivers"
) -> List[InfoCard]:
    """
    Retrieve educational content that caregivers can surface inside the app.

    Example query:
        fetch_health_information("fall prevention exercises")

    The Instant Answer API surfaces abstracts and related topics from trusted
    sources (Wikipedia, reputable medical sites, etc.).
    """
    enriched_query = f"{topic} tips for {audience}"
    payload = _call_instant_answer(enriched_query)
    cards = _cards_from_instant_answer(payload, fallback_title=enriched_query, limit=max_results)

    if not cards:
        cards.append(
            InfoCard(
                title="No guidance found",
                snippet="DuckDuckGo did not return an instant answer for this topic.",
                url="",
            )
        )
    return cards[:max_results]


def locate_emergency_facilities(
    location_query: str, max_results: int = 5
) -> List[InfoCard]:
    """
    Leverage the Instant Answer API to surface experts or organizations that can
    help caregivers find medical support in a specific area.

    While Instant Answer does not expose map coordinates, we can still return
    curated links (e.g., local health departments, emergency preparedness kits,
    senior care organizations) that mention the target location.
    """
    enriched_query = f"{location_query} emergency medical assistance elderly"
    payload = _call_instant_answer(enriched_query)
    cards = _cards_from_instant_answer(payload, fallback_title=enriched_query, limit=max_results)

    if not cards:
        cards.append(
            InfoCard(
                title="No local resources found",
                snippet=(
                    "DuckDuckGo did not return any location-specific instant answers. "
                    "Try broadening the query or specifying a nearby major city."
                ),
                url="",
            )
        )
    return cards[:max_results]


if __name__ == "__main__":
    topic_cards = fetch_health_information("fall prevention checklists")
    resource_cards = locate_emergency_facilities("Kampala, Uganda")

    print("=== Health Information Suggestions ===")
    for card in topic_cards:
        print(f"- {card.title}\n  {card.url}\n  {card.snippet}\n")

    print("=== Regional Medical Resources ===")
    for card in resource_cards:
        meta = card.metadata or {}
        print(f"- {card.title}\n  {card.url}\n  {card.snippet}\n  Source: {meta.get('source', 'N/A')}\n")
