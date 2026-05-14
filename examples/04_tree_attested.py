# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""04 — Tree-attested SHAP for RandomForest / GradientBoosting.

For tree ensembles, CipherExplain runs leaf routing under FHE on the
server while the local SDK runs TreeExplainer on the registered tree
weights. The attestation chain guarantees the server didn't swap
trees mid-flight.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/04_tree_attested.py
"""
import os
import sys

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from cipherexplain_sdk import CipherExplainClient, from_sklearn_ensemble

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=500, n_features=6, n_informative=4,
                            random_state=42)
feature_names = [f"f{i}" for i in range(6)]

# Small RF — n_estimators=30, max_depth=4 fits well within free-tier scope.
rf = RandomForestClassifier(n_estimators=30, max_depth=4,
                              random_state=42).fit(X, y)

spec = from_sklearn_ensemble(rf, "tree_demo", feature_names=feature_names)
try:
    client.delete("tree_demo")
except Exception:
    pass
client.register(spec)
print(f"registered RF: n_estimators={rf.n_estimators}  max_depth={rf.max_depth}")

# Tree-attested explanation.
result = client.explain_tree("tree_demo", X[0].tolist(), model=rf)

ta = result["tree_attested"]
print(f"\nattestation_verified: {ta.attestation_verified}")
print(f"leaves routed under FHE: {len(ta.leaf_indices)}")
print(f"prediction:              {ta.prediction:.4f}")
print(f"base_rate:               {ta.base_rate:.4f}")

print(f"\nSHAP attributions (top 3 by |φ|):")
phis = sorted(zip(feature_names, ta.shap_values),
              key=lambda kv: abs(kv[1]), reverse=True)[:3]
for name, phi in phis:
    print(f"  {name:>6}  {phi:+.4f}")

client.delete("tree_demo")
