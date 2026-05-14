# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""02 — Counterfactual recourse + ECOA Reg-B Form C-1 reason codes.

For a denied applicant, this returns:
  * x_prime          — closest decision-flipping feature vector (minimal-L2)
  * delta            — x_prime - x, ranked by impact
  * reason_codes     — top-4 Form C-1 codes ranked by |φᵢ|
  * decision_flipped — sanity check that x_prime crosses the boundary

The customer's plaintext input never leaves their machine: the server
constructs Enc(x') = Enc(x) ⊕ Enc(δ) homomorphically from session-stored
Enc(x), so the forged-CF attack is closed at the schema layer.

cf_attestation_mode is "UNATTESTED" until v2 ships the encryption-of-
committed-value lattice arm; the Pedersen + π_CF + CRDC + β₂ binding
chain is real today and verifiable.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/02_counterfactual.py
"""
import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import CipherExplainClient, extract_spec

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

# --- 1. Local training (synthetic credit-scoring problem) -----------------
X, y = make_classification(n_samples=600, n_features=6, n_informative=4,
                            class_sep=1.5, random_state=7)
feature_names = [
    "credit_score", "income_proxy", "dti_ratio",
    "employment_months", "existing_debts", "recent_inquiries",
]
model = LogisticRegression(max_iter=1000).fit(X, y)
spec = extract_spec(model, "cf_demo", feature_names=feature_names)

# --- 2. Register ---------------------------------------------------------
try:
    client.delete("cf_demo")
except Exception:
    pass
client.register(spec)

# --- 3. Find a denied applicant ------------------------------------------
denied_idx = next(i for i in range(len(X)) if model.predict([X[i]])[0] == 0)
x_denied = X[denied_idx].tolist()
prob_denied = float(model.predict_proba([x_denied])[0][1])
print(f"applicant {denied_idx}: prob(approve) = {prob_denied:.3f} → DENIED")

# --- 4. Explain --------------------------------------------------------
exp = client.explain_raw("cf_demo", x_denied)

# --- 5. Counterfactual --------------------------------------------------
cf = client.counterfactual("cf_demo", x_denied, exp)

print(f"\ndecision_flipped: {cf['decision_flipped']}")
print(f"cf_attestation_mode: {cf['cf_attestation_mode']}")
print(f"x_prime_origin (metadata): {cf['metadata'].get('x_prime_origin')}")

print(f"\nFeature changes (δ, sorted by |δ|):")
deltas = list(zip(feature_names, cf["delta"]))
for name, d in sorted(deltas, key=lambda kv: abs(kv[1]), reverse=True):
    if abs(d) > 1e-6:
        arrow = "↑" if d > 0 else "↓"
        print(f"  {name:>20}  {arrow} {abs(d):.4f}")

print(f"\nReason codes (Form C-1, ranked by |φ|):")
if cf["reason_codes"]:
    for c in cf["reason_codes"]:
        print(f"  [{c['form_c1_code']}] {c['form_c1_text']}")
        print(f"    feature={c['feature_name']}  |φ|={c['phi_magnitude']:.4f}  "
              f"actionable={c['actionable']}")
else:
    print("  (no codes — synthetic feature names didn't match the bundled "
          "Form C-1 mapping. Use real ECOA feature names — see "
          "example 03 — for live codes.)")

print(f"\nlast warning from server:\n  {cf['warning']}")

# --- Cleanup ------------------------------------------------------------
client.delete("cf_demo")
