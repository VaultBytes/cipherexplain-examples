# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""09 — DP-SHAP: (ε, δ)-differential-privacy noise on published attributions.

For deployments where the SHAP output itself must be differentially
private (e.g. shared dashboards, downstream training), opt in via
``apply_dp=True``. The server's clipped-Gaussian mechanism adds noise
calibrated to a published per-feature sensitivity bound; the
PrivacyAccountant tracks remaining ε budget per key.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/09_dp_shap.py
"""
import os
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import (
    CipherExplainClient, PrivacyAccountant,
    clip_and_noise, compute_sensitivity, extract_spec,
)

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=300, n_features=5, random_state=6)
feature_names = [f"f{i}" for i in range(5)]
model = LogisticRegression(max_iter=1000).fit(X, y)
spec = extract_spec(model, "dp_demo", feature_names=feature_names)

try:
    client.delete("dp_demo")
except Exception:
    pass
client.register(spec)

# --- Server-side DP noise --------------------------------------------------
# Server applies the (ε, δ)-clipped-Gaussian mechanism using
# dp_epsilon, dp_delta, dp_clip_C parameters in the request.
result = client.explain_raw(
    "dp_demo", X[0].tolist(),
    apply_dp=True,
    dp_epsilon=2.0,    # tighter ε = more noise
    dp_delta=1e-5,
    dp_clip_C=0.2,
)

print(f"dp_applied:        {result['dp_applied']}")
print(f"dp_sigma:          {result.get('dp_sigma')}")
print(f"dp_epsilon_used:   {result.get('dp_epsilon_used')}")
print(f"noisy SHAP:        {[f'{v:.4f}' for v in result['shap_values']]}")

# --- DP budget remaining ---------------------------------------------------
usage = client.usage_dp()
print(f"\nDP budget: ε used = {usage.get('epsilon_used', 'n/a')} / "
      f"ε max = {usage.get('epsilon_max', 'n/a')}")
print(f"           remaining = {usage.get('epsilon_remaining', 'n/a')}")

# --- Client-side DP utilities (for offline analysis) -----------------------
# If you'd rather noise SHAP values yourself (e.g. on cached attributions),
# the SDK ships the same mechanism client-side:
sensitivity = compute_sensitivity(num_features=5, clip_norm=0.2)
print(f"\nclient-side L2 sensitivity bound: {sensitivity:.4f}")

# clip_and_noise returns (noisy_shap, sigma) for arbitrary plaintext SHAP.
noisy, sigma = clip_and_noise(
    result["shap_values"], epsilon=1.0, delta=1e-5, clip_norm=0.2,
)
print(f"client-side noisy: {[f'{v:.4f}' for v in noisy]}")
print(f"σ:                {sigma:.4f}")

# PrivacyAccountant tracks composed ε across multiple queries (advanced
# composition theorem, RDP accountant by default).
acct = PrivacyAccountant(epsilon_max=10.0)
for _ in range(3):
    acct.add(epsilon=2.0, delta=1e-5)
print(f"\nlocal accountant after 3 calls: ε spent = {acct.epsilon_spent:.3f} "
      f"of {acct.epsilon_max:.1f}")

client.delete("dp_demo")
