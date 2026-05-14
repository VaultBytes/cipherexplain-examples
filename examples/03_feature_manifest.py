# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""03 — FeatureManifest: immutable + monotone constraints.

ECOA Regulation B forbids using protected-class features (age, sex,
race, ...) as reasons for denial. CipherExplain honours that via
FeatureManifest:

  * immutable[i] = True  → δ_i forced to 0; feature can't appear in
    reason codes either (protection-class scrubbing).
  * monotone_increase[i] = True  → δ_i ≥ 0 (e.g. income can only go
    up — recourse can't ask the applicant to earn less).
  * monotone_decrease[i] = True  → δ_i ≤ 0 (e.g. debt only goes down).

The closed-form δ* projection respects these constraints; if too many
features are blocked, you get a clear ValueError instead of silent
mis-application.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/03_feature_manifest.py
"""
import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import (
    CipherExplainClient, FeatureManifest, extract_spec,
)

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

# Real ECOA-friendly feature names (these match the bundled Form C-1 mapping).
feature_names = [
    "checking_account_status",
    "credit_history",
    "credit_amount",
    "duration_months",
    "savings_account_bonds",
    "age",                 # ← ECOA protection class
    "installment_rate",
    "employment_duration",
    "present_residence",
    "debt_to_income",
]

# Train a synthetic LR on this schema.
X, y = make_classification(n_samples=400, n_features=10, n_informative=7,
                            random_state=11)
model = LogisticRegression(max_iter=1000).fit(X, y)
spec = extract_spec(model, "manifest_demo", feature_names=feature_names)

try:
    client.delete("manifest_demo")
except Exception:
    pass
client.register(spec)

# Find a denied applicant.
denied_idx = next(i for i in range(len(X)) if model.predict([X[i]])[0] == 0)
x_denied = X[denied_idx].tolist()
exp = client.explain_raw("manifest_demo", x_denied)

# Manifest:
#   age (idx 5)           → immutable (ECOA protected)
#   credit_amount (idx 2) → monotone_decrease (recourse can't *raise* debt)
#   duration_months (idx 3) → monotone_decrease
manifest = FeatureManifest(
    immutable=[
        False, False, False, False, False,
        True,                 # age
        False, False, False, False,
    ],
    monotone_decrease=[
        False, False, True,   # credit_amount: can't go up
        True,                 # duration_months: can't go up
        False, False, False, False, False, False,
    ],
)

cf = client.counterfactual("manifest_demo", x_denied, exp,
                            feature_manifest=manifest)

# Verify constraints
assert cf["delta"][5] == 0.0, "age must be immutable"
assert cf["delta"][2] <= 0.0, "credit_amount must not increase"
assert cf["delta"][3] <= 0.0, "duration_months must not increase"
print("constraints OK: age unchanged, credit_amount + duration non-increasing")

# age cannot appear in reason codes.
code_features = [c["feature_name"] for c in cf["reason_codes"]]
assert "age" not in code_features, "ECOA: age leaked into reason codes"
print("ECOA protection-class scrub OK: 'age' not in reason codes")

print(f"\nFeasible recourse (constraint-aware):")
for i, (name, d) in enumerate(zip(feature_names, cf["delta"])):
    if abs(d) > 1e-6:
        arrow = "↑" if d > 0 else "↓"
        sign_info = ""
        if manifest.immutable and manifest.immutable[i]:
            sign_info = "  [IMMUTABLE]"
        elif manifest.monotone_decrease and manifest.monotone_decrease[i]:
            sign_info = "  [must-decrease]"
        elif manifest.monotone_increase and manifest.monotone_increase[i]:
            sign_info = "  [must-increase]"
        print(f"  {name:>26}  {arrow} {abs(d):.4f}{sign_info}")

print(f"\ndecision_flipped: {cf['decision_flipped']}")
print(f"reason codes returned: {len(cf['reason_codes'])}")

client.delete("manifest_demo")
