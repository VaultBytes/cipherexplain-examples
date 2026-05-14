# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""11 — API key rotation + model versioning.

Two ops features every regulated deployment needs:

  * client.rotate_key() — atomically swap the API key. The old one is
    deactivated server-side as soon as the new one is returned. Useful
    for secret rotation pipelines (every 30/90 days).

  * Re-register the same model_id with new weights — the server stamps
    a fresh model_version_id. Every /explain response carries the
    version_id used, so compliance reports pin exactly which weights
    produced an attribution.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/11_key_rotation_versioning.py
"""
import os
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import CipherExplainClient, extract_spec

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=200, n_features=4, random_state=8)
feature_names = [f"f{i}" for i in range(4)]

# --- Register v1 ----------------------------------------------------------
model_v1 = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
try:
    client.delete("version_demo")
except Exception:
    pass
client.register(extract_spec(model_v1, "version_demo", feature_names=feature_names))
r1 = client.explain_raw("version_demo", X[0].tolist())
print(f"v1: model_version={r1['model_version']!r} "
      f"version_id={r1.get('model_version_id', '')[:8]}...")

# --- Re-register with new weights → v2 -----------------------------------
model_v2 = LogisticRegression(max_iter=1000, C=0.1).fit(X, y)  # different reg
client.register(extract_spec(model_v2, "version_demo",
                              feature_names=feature_names))
r2 = client.explain_raw("version_demo", X[0].tolist())
print(f"v2: model_version={r2['model_version']!r} "
      f"version_id={r2.get('model_version_id', '')[:8]}...")

assert r1.get("model_version_id") != r2.get("model_version_id"), (
    "version_id should differ after re-registration"
)
print("version_id changed across re-registration — audit trail OK")

# --- Key rotation ---------------------------------------------------------
# WARNING: this DEACTIVATES the current key. Save the new key before
# any other call. In real deployments wrap this in a transaction with
# your secret manager (AWS Secrets Manager, Vault, etc.).
print(f"\nrotating API key (old key will deactivate immediately)...")
rotation = client.rotate_key()
new_key = rotation["new_key"]
print(f"new key: {new_key[:8]}... (save this)")
print(f"old key now: {'inactive' if rotation.get('old_deactivated') else 'unknown'}")

# Use the new key for the next call.
client_new = CipherExplainClient(api_key=new_key)
r3 = client_new.explain_raw("version_demo", X[0].tolist())
print(f"call with new key OK: prediction={r3['prediction']:.4f}")

# Old client should now fail (key deactivated).
try:
    client.explain_raw("version_demo", X[0].tolist())
    print("WARN: old key still works (expected 401/403)")
except Exception as exc:
    print(f"old key correctly rejected: {type(exc).__name__}")

client_new.delete("version_demo")

print(f"\nIMPORTANT: SAVE THE NEW KEY → {new_key}")
print("(this script printed it once; we can't recover it server-side.)")
