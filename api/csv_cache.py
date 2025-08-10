import csv
import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_directory_exists(directory_path: str) -> None:
    if directory_path and not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def _stable_params_hash(params: Optional[Dict[str, Any]]) -> str:
    canonical = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _current_unix_timestamp() -> int:
    return int(time.time())


@dataclass(frozen=True)
class CacheKey:
    source: str
    params_hash: str

    @staticmethod
    def from_source_and_params(source: str, params: Optional[Dict[str, Any]]) -> "CacheKey":
        return CacheKey(source=source, params_hash=_stable_params_hash(params))

    def as_string(self) -> str:
        return f"{self.source}:{self.params_hash}"


class CsvCache:
    """
    File-backed cache that stores entries in CSV files.

    Each logical dataset (e.g., an API you call) can be stored in its own CSV file
    identified by table_name. Rows are keyed by a computed string key which is the
    combination of `source` and a stable hash of request params.

    CSV columns:
      - key: string (source + params hash)
      - source: string (human-friendly identifier)
      - params_json: canonical JSON string of parameters
      - data_json: JSON-serialized response payload
      - timestamp: UNIX epoch seconds when cached
    """

    def __init__(self, cache_directory: Optional[str] = None) -> None:
        """
        cache_directory:
          - None: defaults to a standard location inside the API package: api/cache
          - relative path: resolved relative to api/ (module) directory
          - absolute path: used as-is
        """
        if cache_directory is None:
            self.cache_directory = os.path.join(MODULE_DIR, "cache")
        else:
            self.cache_directory = (
                cache_directory
                if os.path.isabs(cache_directory)
                else os.path.join(MODULE_DIR, cache_directory)
            )
        _ensure_directory_exists(self.cache_directory)

    def _file_path(self, table_name: str) -> str:
        """
        Build a file path anchored under self.cache_directory.
        Supports subfolders via slashes in table_name, e.g., "alphavantage/news".
        The final file will be <cache_dir>/<table_name>.csv
        """
        # Normalize relative subpath and prevent escaping the base directory
        relative = table_name.strip()
        if not relative:
            relative = "default"
        filename = relative if relative.endswith(".csv") else f"{relative}.csv"
        full_path = os.path.normpath(os.path.join(self.cache_directory, filename))
        base = os.path.normpath(self.cache_directory)
        # Ensure the result stays within base directory
        if os.path.commonpath([full_path, base]) != base:
            raise ValueError("Invalid table_name resulting in path traversal")
        # Ensure parent directories exist
        _ensure_directory_exists(os.path.dirname(full_path))
        return full_path

    def _read_rows_by_key(self, table_name: str) -> Dict[str, Dict[str, str]]:
        file_path = self._file_path(table_name)
        if not os.path.exists(file_path):
            return {}
        with open(file_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows: Dict[str, Dict[str, str]] = {}
            for row in reader:
                key = row.get("key")
                if not key:
                    continue
                rows[key] = row
            return rows

    def _write_rows(self, table_name: str, rows: Sequence[Dict[str, str]]) -> None:
        file_path = self._file_path(table_name)
        fieldnames = ["key", "source", "params_json", "data_json", "timestamp"]
        # Write file atomically to avoid partial writes
        temp_path = file_path + ".tmp"
        with open(temp_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        os.replace(temp_path, file_path)

    def _append_row(self, table_name: str, row: Dict[str, str]) -> None:
        file_path = self._file_path(table_name)
        exists = os.path.exists(file_path)
        fieldnames = ["key", "source", "params_json", "data_json", "timestamp"]
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    def get(
        self,
        table_name: str,
        source: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[Any]:
        rows = self._read_rows_by_key(table_name)
        cache_key = CacheKey.from_source_and_params(source, params)
        row = rows.get(cache_key.as_string())
        if not row:
            return None

        if ttl_seconds is not None:
            try:
                ts = int(row.get("timestamp", "0"))
            except ValueError:
                ts = 0
            if _current_unix_timestamp() - ts > ttl_seconds:
                return None

        try:
            return json.loads(row.get("data_json", "null"))
        except json.JSONDecodeError:
            return None

    def set(
        self,
        table_name: str,
        source: str,
        params: Optional[Dict[str, Any]],
        data: Any,
    ) -> None:
        cache_key = CacheKey.from_source_and_params(source, params)
        # Read all rows and upsert to avoid duplicates
        rows_by_key = self._read_rows_by_key(table_name)
        row = {
            "key": cache_key.as_string(),
            "source": source,
            "params_json": json.dumps(params or {}, sort_keys=True, separators=(",", ":")),
            "data_json": json.dumps(data, separators=(",", ":")),
            "timestamp": str(_current_unix_timestamp()),
        }
        rows_by_key[cache_key.as_string()] = row
        # Persist rows back
        self._write_rows(table_name, list(rows_by_key.values()))

    def get_or_fetch(
        self,
        table_name: str,
        source: str,
        params: Optional[Dict[str, Any]],
        fetch_fn: Callable[[], Any],
        ttl_seconds: Optional[int] = None,
    ) -> Tuple[Any, bool]:
        """
        Returns (data, from_cache_flag)
        """
        cached = self.get(table_name, source, params, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached, True

        data = fetch_fn()
        self.set(table_name, source, params, data)
        return data, False

    def get_or_fetch_many(
        self,
        table_name: str,
        items: Sequence[Tuple[str, Optional[Dict[str, Any]]]],
        fetch_missing_fn: Callable[[List[Tuple[str, Optional[Dict[str, Any]]]]], Dict[str, Any]],
        ttl_seconds: Optional[int] = None,
    ) -> Tuple[List[Any], List[bool]]:
        """
        Batch variant that:
          - checks cache for each (source, params)
          - fetches all missing in one call via fetch_missing_fn(missing_items)
          - writes new entries

        Returns (results_in_input_order, from_cache_flags_in_input_order)
        """
        results: List[Any] = []
        from_cache_flags: List[bool] = []

        # Build lookup for cached items
        rows_by_key = self._read_rows_by_key(table_name)
        keys: List[CacheKey] = [CacheKey.from_source_and_params(src, prm) for src, prm in items]

        missing_items: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        missing_indices: List[int] = []

        for idx, (src, prm) in enumerate(items):
            ck = CacheKey.from_source_and_params(src, prm)
            row = rows_by_key.get(ck.as_string())
            use_cached = False
            data_value: Any = None

            if row:
                if ttl_seconds is not None:
                    try:
                        ts = int(row.get("timestamp", "0"))
                    except ValueError:
                        ts = 0
                    if _current_unix_timestamp() - ts <= ttl_seconds:
                        try:
                            data_value = json.loads(row.get("data_json", "null"))
                            use_cached = True
                        except json.JSONDecodeError:
                            use_cached = False
                else:
                    try:
                        data_value = json.loads(row.get("data_json", "null"))
                        use_cached = True
                    except json.JSONDecodeError:
                        use_cached = False

            if use_cached:
                results.append(data_value)
                from_cache_flags.append(True)
            else:
                results.append(None)  # placeholder
                from_cache_flags.append(False)
                missing_items.append((src, prm))
                missing_indices.append(idx)

        if missing_items:
            fetched_map = fetch_missing_fn(missing_items)
            # fetched_map must map a compound string key to data or just source->data for no-params use cases
            for mi, idx in zip(missing_items, missing_indices):
                src, prm = mi
                ck = CacheKey.from_source_and_params(src, prm)
                # Prefer exact compound key match; fallback to source-only
                compound_key = ck.as_string()
                data_value = fetched_map.get(compound_key)
                if data_value is None:
                    data_value = fetched_map.get(src)
                # Persist and fill result
                self.set(table_name, src, prm, data_value)
                results[idx] = data_value

        return results, from_cache_flags


