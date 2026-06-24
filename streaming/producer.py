"""
streaming/producer.py

Kafka producer that generates synthetic AML transaction events and
publishes them to the "aml_transactions" topic at a configurable rate.

Transaction schema mirrors scripts/ingest_transactions.py exactly so that
streaming/structured_streaming_consumer.py can apply the same typology logic.

Usage:
    pip install confluent-kafka
    python streaming/producer.py

Environment variables:
    KAFKA_BOOTSTRAP_SERVERS   default: localhost:9092
    PRODUCER_INTERVAL         seconds between messages, default: 0.5
"""

import json
import os
import random
import time
from datetime import datetime, timedelta

from confluent_kafka import Producer

# ── Configuration ──────────────────────────────────────────────────────────────
TOPIC             = "aml_transactions"
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
PRODUCER_INTERVAL = float(os.environ.get("PRODUCER_INTERVAL", "0.5"))

# ── Synthetic data pools (match ingest_transactions.py) ───────────────────────
N_CUSTOMERS = 500
N_ACCOUNTS  = 600

CURRENCIES   = ["EUR", "USD", "GBP", "SEK", "CHF"]
CHANNELS     = ["SEPA", "SWIFT", "FASTER_PAYMENTS", "INTERNAL", "CARD"]
COUNTRY_POOL = ["LV", "EE", "LT", "FI", "SE", "DE", "GB", "CY", "MT", "PAN", "BVI"]
LOW_RISK_COUNTRIES = ["LV", "EE", "LT", "FI", "SE", "DE", "GB"]

customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(N_CUSTOMERS)]
account_ids  = [f"ACC{str(i).zfill(6)}"  for i in range(N_ACCOUNTS)]

_txn_counter = 0


def delivery_report(err, msg):
    if err is not None:
        print(f"[ERROR] Delivery failed for {msg.topic()}/{msg.partition()}: {err}")


def random_amount(suspicious: bool = False) -> float:
    """Near-threshold amounts for suspicious transactions, exponential otherwise."""
    if suspicious:
        return round(random.uniform(9_000, 9_999), 2)
    return round(random.expovariate(1 / 2500), 2)


def make_transaction(txn_id: int) -> dict:
    """Generate one synthetic transaction matching the batch pipeline schema."""
    suspicious = random.random() < 0.08
    # Use recent timestamps so streaming windows see relevant data
    ts = datetime.utcnow() - timedelta(seconds=random.randint(0, 30))
    return {
        "transaction_id":      f"TXN{str(txn_id).zfill(8)}",
        "customer_id":         random.choice(customer_ids),
        "originator_account":  random.choice(account_ids),
        "beneficiary_account": random.choice(account_ids),
        "amount":              random_amount(suspicious),
        "currency":            random.choice(CURRENCIES),
        "channel":             random.choice(CHANNELS),
        "originator_country":  random.choice(COUNTRY_POOL),
        "beneficiary_country": (
            random.choice(COUNTRY_POOL) if suspicious
            else random.choice(LOW_RISK_COUNTRIES)
        ),
        "timestamp":        ts.isoformat(),
        "is_flagged_source": suspicious,
    }


def main():
    global _txn_counter

    conf = {"bootstrap.servers": BOOTSTRAP_SERVERS}
    producer = Producer(conf)

    print(f"Producer started. Publishing to '{TOPIC}' on {BOOTSTRAP_SERVERS}")
    print(f"Interval: {PRODUCER_INTERVAL}s  |  Press Ctrl+C to stop.")

    try:
        while True:
            txn = make_transaction(_txn_counter)
            _txn_counter += 1

            payload = json.dumps(txn).encode("utf-8")
            producer.produce(
                topic=TOPIC,
                key=txn["customer_id"].encode("utf-8"),
                value=payload,
                on_delivery=delivery_report,
            )
            # Poll to trigger delivery callbacks without blocking
            producer.poll(0)

            print(
                f"Sent txn_id={txn['transaction_id']} "
                f"amount={txn['amount']:.2f} "
                f"to {TOPIC}"
            )
            time.sleep(PRODUCER_INTERVAL)

    except KeyboardInterrupt:
        print("\nShutting down producer...")
    finally:
        remaining = producer.flush(timeout=10)
        if remaining:
            print(f"[WARN] {remaining} message(s) not delivered before shutdown.")
        print("Producer stopped.")


if __name__ == "__main__":
    main()
