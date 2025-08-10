from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from csv_cache import CsvCache, CacheKey


def fake_api_call(params: Dict[str, Any]) -> Dict[str, Any]:
    # Simulate an expensive call
    time.sleep(0.5)
    return {
        "params": params,
        "value": random.randint(1, 1_000_000),
        "fetched_at": int(time.time()),
    }


def fake_batch_api_call(items: Sequence[Tuple[str, Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
    # Simulate a batch API: returns a dict mapping compound cache key to data
    time.sleep(0.7)
    result: Dict[str, Any] = {}
    for source, params in items:
        result[f"{source}:{CsvCache._CsvCache__debug_params_hash_for_example(params)}"] = {
            "source": source,
            "params": params or {},
            "value": random.randint(1, 1_000_000),
            "fetched_at": int(time.time()),
        }
    return result


def main() -> None:
    cache = CsvCache(cache_directory="cache")
    table = "demo"

    # Single get-or-fetch
    params = {"q": "socks", "page": 1}
    data, from_cache = cache.get_or_fetch(
        table_name=table,
        source="search",
        params=params,
        fetch_fn=lambda: fake_api_call(params),
        ttl_seconds=60,  # cache entries older than 60s are re-fetched
    )
    print("single:", {"from_cache": from_cache, "data": data})

    # Repeat immediately; should hit cache
    data2, from_cache2 = cache.get_or_fetch(
        table_name=table,
        source="search",
        params=params,
        fetch_fn=lambda: fake_api_call(params),
        ttl_seconds=60,
    )
    print("single_repeat:", {"from_cache": from_cache2})

    # Batch example
    items: List[Tuple[str, Optional[Dict[str, Any]]]] = [
        ("product", {"id": 1}),
        ("product", {"id": 2}),
        ("product", {"id": 3}),
    ]

    def batch_fetch(missing_items: Sequence[Tuple[str, Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
        # Map to the compound key used internally: source:hash(params)
        # We reconstruct the same key format used in CsvCache
        out: Dict[str, Any] = {}
        for source, params in missing_items:
            data = fake_api_call(params or {})
            # Mirror CsvCache compound key format
            from hashlib import sha256
            import json

            canonical = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
            params_hash = sha256(canonical.encode("utf-8")).hexdigest()
            out[f"{source}:{params_hash}"] = data
        return out

    results, flags = cache.get_or_fetch_many(
        table_name=table,
        items=items,
        fetch_missing_fn=batch_fetch,
        ttl_seconds=60,
    )
    print("batch:", {"from_cache_flags": flags})


if __name__ == "__main__":
    main()


