# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""08 — Async batch explain with webhook callback.

For compliance workflows where you need to score thousands of records
overnight, ``explain_batch`` queues the job and POSTs results to your
webhook URL when each row finishes. No long-poll, no holding HTTP
connections open.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/08_async_batch.py
"""
import os
import sys
import time

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import CipherExplainClient, extract_spec

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=200, n_features=4, random_state=5)
feature_names = [f"f{i}" for i in range(4)]
model = LogisticRegression(max_iter=1000).fit(X, y)
spec = extract_spec(model, "batch_demo", feature_names=feature_names)

try:
    client.delete("batch_demo")
except Exception:
    pass
client.register(spec)

# Set CIPHEREXPLAIN_WEBHOOK_URL to your own endpoint to receive results.
webhook = os.environ.get("CIPHEREXPLAIN_WEBHOOK_URL",
                          "https://webhook.site/test-cipherexplain")
print(f"webhook URL: {webhook}")

# Queue 5 rows.
rows = X[:5].tolist()
job = client.explain_batch(rows, model_id="batch_demo",
                            webhook_url=webhook)
print(f"job_id: {job['job_id']}  status: {job.get('status', 'queued')}")

# Poll status (no webhook hit, no waiting). 5 LR rows finish in seconds.
for _ in range(20):
    status = client.explain_batch_status(job["job_id"])
    print(f"  status: {status['status']}  "
          f"completed: {status.get('completed_count', 0)}/{status.get('total_count', 0)}")
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(2)

# Results are also fetchable directly (in addition to webhook delivery).
results = client.explain_batch_results(job["job_id"])
print(f"\nfetched {len(results)} results")
for i, r in enumerate(results[:2]):
    print(f"  row {i}: prediction={r['prediction']:.3f}  "
          f"top-feature={feature_names[max(range(4), key=lambda k: abs(r['shap_values'][k]))]}")

client.delete("batch_demo")
