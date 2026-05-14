# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""05 — sklearn Pipeline: scaler auto-unwrapped server-side.

If you train ``Pipeline([("scaler", StandardScaler()), ("clf", LR())])``,
register_pipeline() ships the scaler params + classifier weights as a
single spec. ``explain_raw()`` then accepts RAW (un-scaled) feature
values; the server scales them with the same params before the SHAP
computation. Eliminates the "did I scale this right?" failure mode.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/05_sklearn_pipeline.py
"""
import os
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cipherexplain_sdk import CipherExplainClient

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=400, n_features=5, n_informative=4,
                            random_state=2)
feature_names = ["income", "credit_score", "dti", "age", "tenure"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
]).fit(X, y)

try:
    client.delete("pipeline_demo")
except Exception:
    pass
client.register_pipeline("pipeline_demo", pipe, feature_names=feature_names,
                         X_train=X)
print("registered Pipeline(scaler, LR) — scaler params shipped, not pickled")

# Raw values (un-scaled).
x_raw = X[0].tolist()
result = client.explain_raw("pipeline_demo", x_raw)

print(f"\nprediction (server applied scaler): {result['prediction']:.4f}")
print(f"base_rate:                          {result['base_rate']:.4f}")
print(f"SHAP attributions (in raw-feature units):")
for n, phi in sorted(zip(feature_names, result["shap_values"]),
                      key=lambda kv: abs(kv[1]), reverse=True):
    print(f"  {n:>14}  {phi:+.4f}")

client.delete("pipeline_demo")
