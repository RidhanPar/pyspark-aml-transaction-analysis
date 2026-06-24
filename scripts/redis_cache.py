"""
scripts/redis_cache.py

RedisCache: thin wrapper around redis-py for the AML pipeline.

Two responsibilities:
  1. Country risk reference cache — stores the HIGH_RISK_COUNTRIES set as a
     Redis SET (key: aml:high_risk_countries) with a 1-hour TTL so that
     dynamic country-list updates require no code changes.
  2. Pipeline run observability — stores lightweight run metadata as a Redis
     HASH (key: aml:pipeline_runs:{run_id}) with a 24-hour TTL.

The pipeline driver calls populate_country_risk_cache() then broadcasts the
result to Spark workers via SparkContext.broadcast(); workers never connect to
Redis directly.

Environment variables:
    REDIS_HOST   default: localhost
    REDIS_PORT   default: 6379
"""

import os
from datetime import datetime

import redis


class RedisCache:

    def __init__(self):
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        print(f"[Redis] Connecting to {host}:{port} …")
        self.client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
        )
        try:
            self.client.ping()
            print(f"[Redis] Connected to {host}:{port}")
        except redis.exceptions.ConnectionError as exc:
            print(f"[Redis] Connection failed: {exc}")

    # ── Country risk reference cache ───────────────────────────────────────────

    def populate_country_risk_cache(self, countries: list) -> int:
        """
        Loads country codes into the Redis SET aml:high_risk_countries.
        Deletes any existing key first so the SET always reflects the
        current list. Sets a 1-hour TTL.

        Returns the number of members added.
        """
        self.client.delete("aml:high_risk_countries")
        count = self.client.sadd("aml:high_risk_countries", *countries)
        self.client.expire("aml:high_risk_countries", 3600)
        print(f"[Redis] Country risk cache populated: {count} countries, TTL=3600s")
        return count

    def get_high_risk_countries(self) -> set:
        """
        Returns the cached high-risk country codes as a Python set.
        Returns an empty set (and logs a warning) if the key is missing
        or the TTL has expired.
        """
        members = self.client.smembers("aml:high_risk_countries")
        if not members:
            print(
                "[WARN][Redis] aml:high_risk_countries is empty or missing "
                "— TTL may have expired; caller should fall back to hardcoded list"
            )
            return set()
        return set(members)

    # ── Pipeline run observability ─────────────────────────────────────────────

    def store_pipeline_run(
        self,
        run_id: str,
        record_count: int,
        alert_count: int,
        status: str = "completed",
    ) -> None:
        """
        Stores lightweight pipeline run metadata as a Redis HASH with a
        24-hour TTL. Fields: record_count, alert_count, status, timestamp.
        """
        key = f"aml:pipeline_runs:{run_id}"
        self.client.hset(
            key,
            mapping={
                "record_count": record_count,
                "alert_count":  alert_count,
                "status":       status,
                "timestamp":    datetime.utcnow().isoformat(),
            },
        )
        self.client.expire(key, 86400)
        print(
            f"[Redis] Pipeline run {run_id} stored: "
            f"{record_count} records, {alert_count} alerts"
        )

    def get_pipeline_run(self, run_id: str) -> dict:
        """Returns all fields of a stored pipeline run, or an empty dict."""
        return self.client.hgetall(f"aml:pipeline_runs:{run_id}")

    # ── Health ─────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Returns True if Redis responds to PING, False on any exception."""
        try:
            return bool(self.client.ping())
        except Exception:
            return False
