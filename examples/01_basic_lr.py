# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""01 — Basic logistic regression: register weights, get encrypted SHAP.

The simplest possible CipherExplain flow:

  1. Train an LR locally (nothing leaves your machine).
  2. Extract just the weights — no pickle, no training data.
  3. Register them under a model_id.
  4. Get per-feature SHAP attributions for a single input.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/01_basic_lr.py
"""
import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import CipherExplainClient, extract_spec

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY")
if not API_KEY:
    sys.exit("Set CIPHEREXPLAIN_API_KEY (get a free key at vaultbytes.com/cipherexplain)")

# --- 1. Local training -----------------------------------------------------
X, y = make_classification(n_samples=500, n_features=8, n_informative=5,
                            random_state=42)
feature_names = [f"feature_{i}" for i in range(8)]
model = LogisticRegression(max_iter=1000).fit(X, y)
print(f"trained LR on {X.shape[0]} samples × {X.shape[1]} features")

# --- 2. Extract weights ---------------------------------------------------
spec = extract_spec(model, "demo_lr", feature_names=feature_names)
print(f"spec: model_type={spec['model_type']}  d={len(feature_names)}")

# --- 3. Register ----------------------------------------------------------
client = CipherExplainClient(api_key=API_KEY)
try:
    client.delete("demo_lr")
except Exception:
    pass  # not registered yet — fine
client.register(spec)
print("registered as 'demo_lr'")

# --- 4. Explain -----------------------------------------------------------
x_query = X[0].tolist()  # explain the first row
result = client.explain_raw("demo_lr", x_query)

print(f"\nprediction: {result['prediction']:.4f}")
print(f"base rate:  {result['base_rate']:.4f}")
print("\nSHAP attributions (highest |φ| first):")
phis = list(zip(feature_names, result["shap_values"]))
for name, phi in sorted(phis, key=lambda kv: abs(kv[1]), reverse=True):
    bar = "█" * int(abs(phi) * 50)
    print(f"  {name:>12}  {phi:+.4f}  {bar}")

print(f"\nmetadata.fhe_mode_used: {result.get('fhe_mode_used')}")
print(f"sum(shap) + base_rate ≈ prediction: "
      f"{sum(result['shap_values']) + result['base_rate']:.4f}")

# --- Cleanup --------------------------------------------------------------
client.delete("demo_lr")
