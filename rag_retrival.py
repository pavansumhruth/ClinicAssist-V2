import requests
from typing import Tuple, Dict, Any

from config import SEARCH_API_BASE

def _clinical_history_chunk(patient_id: str, query: str) -> Tuple[Dict[str, Any], list, dict]:
    try:
        url = f"{SEARCH_API_BASE}/search"
        print(f"[DEBUG] Calling search API: {url} with patient_id={patient_id}, query={query}")

        response = requests.get(
            url,
            params={
                "patient_id": patient_id,
                "query": query
            },
            timeout=10
        )

        print(f"[DEBUG] Search API status code: {response.status_code}")

        if response.status_code == 404:
            print("[DEBUG] No search results found — using empty fallback")
            return {}, [], []

        response.raise_for_status()

        data = response.json()
        context = data.get("context", [])

        print(f"[DEBUG] Total chunks received: {len(context)}")

        # ✅ Get chunk with least distance
        best_chunk = min(context, key=lambda x: x.get("distance", float("inf"))) if context else {}

        if best_chunk:
            print(f"[DEBUG] Best chunk distance: {best_chunk.get('distance')}")
            print(f"[DEBUG] Best chunk type: {best_chunk.get('chunk_type')}")

        return data, context, best_chunk.get("text", []) if isinstance(best_chunk, dict) else []

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Search API request failed: {str(e)}")
        return {}, [], []

#data, context, best_chunk = _clinical_history_chunk(
#    patient_id="791427b4-9cc4-8bcc-3fee-e3e14b6d3fea",
#    query="miscarage"
#)

#print("BEST MATCH:")
#print(best_chunk)

