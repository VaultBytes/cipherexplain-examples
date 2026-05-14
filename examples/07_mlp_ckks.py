# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""07 — MLP (ReLU, 2-3 hidden layers) via full CKKS homomorphic evaluation.

For MLP models the server evaluates every coalition under CKKS;
client-side decrypt happens only on the final SHAP vector. Latency is
higher (~70s on prod CX22) but the entire forward pass is encrypted.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/07_mlp_ckks.py
"""
import os
import sys
import time

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

from cipherexplain_sdk import CipherExplainClient

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                            random_state=4)
feature_names = [f"f{i}" for i in range(6)]

mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                     random_state=4).fit(X, y)

try:
    client.delete("mlp_demo")
except Exception:
    pass
client.register_mlp("mlp_demo", mlp, feature_names=feature_names, X_train=X)
print(f"registered MLP: hidden={mlp.hidden_layer_sizes}  layers={len(mlp.coefs_)}")

# fhe_mode='execute' → real CKKS evaluation (slower but encrypted).
print("\nrunning under CKKS — this can take ~70s on the free tier...")
t0 = time.perf_counter()
result = client.explain_raw("mlp_demo", X[0].tolist(),
                              fhe_mode="execute")
elapsed = time.perf_counter() - t0
print(f"elapsed: {elapsed:.1f} s")

print(f"\nfhe_mode_requested:  {result.get('fhe_mode_requested')}")
print(f"fhe_mode_used:       {result.get('fhe_mode_used')}")
print(f"prediction:          {result['prediction']:.4f}")
print(f"base_rate:           {result['base_rate']:.4f}")
print(f"shap_values length:  {len(result['shap_values'])}")

# The MLP path can opt into the linear-surrogate fast lane (~7 s) if the
# caller's accuracy budget tolerates an L∞ error bound of 0.15:
#   result = client.explain_raw("mlp_demo", x, fhe_mode="execute",
#                                 linear_surrogate=True)

client.delete("mlp_demo")
