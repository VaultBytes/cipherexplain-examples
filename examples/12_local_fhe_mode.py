# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""12 — Local CKKS encryption (fhe_mode='ckks').

For the strictest privacy posture: the SDK encrypts your input
client-side under your own keys, the server runs CKKS homomorphic
computation without ever decrypting, and the SDK decrypts the result.
Requires the [fhe] extra (openfhe).

  pip install 'cipherexplain[fhe]'

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/12_local_fhe_mode.py
"""
import os
import sys

try:
    import openfhe  # noqa: F401
except ImportError:
    sys.exit("This example requires the [fhe] extra: "
              "pip install 'cipherexplain[fhe]'")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import (
    CipherExplainClient, FHEUnavailableError, extract_spec,
)

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=200, n_features=5, random_state=9)
feature_names = [f"f{i}" for i in range(5)]
model = LogisticRegression(max_iter=1000).fit(X, y)

try:
    client.delete("ckks_demo")
except Exception:
    pass
client.register(extract_spec(model, "ckks_demo",
                              feature_names=feature_names))

# fhe_mode='ckks' enables client-side encryption. The SDK creates a
# throwaway CKKS keypair, encrypts X[0] locally, sends ciphertext to
# the server, server evaluates SHAP on the ciphertext, returns
# encrypted result, SDK decrypts locally. The server NEVER sees
# plaintext input or output.
try:
    result = client.explain_raw("ckks_demo", X[0].tolist(),
                                  fhe_mode="ckks")
except FHEUnavailableError as exc:
    sys.exit(f"FHE unavailable on this server: {exc}")

print(f"fhe_mode_requested: {result['fhe_mode_requested']}")
print(f"fhe_mode_used:      {result['fhe_mode_used']}")
print(f"prediction:         {result['prediction']:.4f}")
print(f"SHAP (locally decrypted):")
for n, phi in sorted(zip(feature_names, result["shap_values"]),
                      key=lambda kv: abs(kv[1]), reverse=True):
    print(f"  {n:>4}  {phi:+.4f}")

# fhe_mode_strict=True turns any silent downgrade into a 400 error so
# you never serve a plaintext explanation while believing it was
# encrypted:
#   result = client.explain_raw("ckks_demo", X[0].tolist(),
#                                fhe_mode="ckks", fhe_mode_strict=True)

client.delete("ckks_demo")
