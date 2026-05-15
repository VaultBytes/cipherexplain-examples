# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""13 — v1-B lattice attestation: bgv_zk=True for B2B internal attestation.

This is the cryptographic upgrade that lands the response's
`cf_attestation_mode` field from `"UNATTESTED"` to `"ATTESTED_BGV_ZK"`.

What it does, in one line
-------------------------
Adds a ~16.5 KB del Pino (PKC 2019) lattice zero-knowledge proof to the
counterfactual request. The SDK generates a fresh BGV keypair locally,
encrypts the integer-quantised SHAP attribution vector a second time
(parallel to the existing CKKS ciphertext), and proves to the server
that the BGV ciphertext encodes the same value committed in the
existing Pedersen commitment on BLS12-381 G1.

The server NEVER decrypts the BGV ciphertext — only the public proof
math runs on the server. Verifying takes ~50 ms.

What this proves
----------------
The server cannot fabricate a SHAP output that contradicts its own
FHE computation on the input. An auditor reading the response can
verify, after the fact, that the explanation the customer received
corresponded to a specific ciphertext under a specific BGV public key.

Honest-client caveat (READ BEFORE DEPLOYING)
--------------------------------------------
The proof binds the BGV ciphertext to the response, but it does NOT
cryptographically bind the BGV ciphertext to the CKKS ciphertext that
the SHAP computation actually runs on. Proving cross-encryption-scheme
consistency is provably impossible without decryption (scheme-switching
hardness, eprint 2023/988). So a malicious **client** could submit
inconsistent pairs.

This is fine for B2B internal attestation (a bank attests its own SHAP
to a third-party auditor — the bank has no incentive to cheat against
itself). It is NOT fine for consumer-facing ECOA/Reg-B adverse-action
use cases — there, stick with the default `bgv_zk=False`.

Server-side gating
------------------
The server only honours `bgv_zk=True` payloads when
`CE_CF_USE_BGV_ZK=1` is set on the deployment. Until then, the request
is accepted but the v1-B branch is skipped and the response stays on
the v1-A path (mode `"UNATTESTED"`). This lets SDKs roll out v0.6.0 to
PyPI without coordinated server changes; the customer-visible flip
happens when the operator sets the env var.

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/13_bgv_zk_b2b_attestation.py

Requires:
  pip install 'cipherexplain[lattice]>=0.6.0'
"""
import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import CipherExplainClient, extract_spec

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY"
)
client = CipherExplainClient(api_key=API_KEY)

# --- 1. Local training (same synthetic problem as example 02) -----------
X, y = make_classification(
    n_samples=600, n_features=6, n_informative=4, class_sep=1.5, random_state=7
)
feature_names = [
    "credit_score",
    "income_proxy",
    "dti_ratio",
    "employment_months",
    "existing_debts",
    "recent_inquiries",
]
model = LogisticRegression(max_iter=1000).fit(X, y)
spec = extract_spec(model, "bgv_zk_demo", feature_names=feature_names)

# --- 2. Register --------------------------------------------------------
try:
    client.delete("bgv_zk_demo")
except Exception:
    pass
client.register(spec)

# --- 3. Find a denied applicant -----------------------------------------
denied_idx = next(i for i in range(len(X)) if model.predict([X[i]])[0] == 0)
x_denied = X[denied_idx].tolist()
print(f"applicant {denied_idx}: DENIED — running counterfactual + lattice proof")

# --- 4. Explain (server-side encrypted SHAP) ----------------------------
exp = client.explain_raw("bgv_zk_demo", x_denied)

# --- 5. Counterfactual with v1-B lattice attestation --------------------
# This is the only line that changes vs example 02_counterfactual.py.
# The SDK adds ~16.5 KB to the request (the lattice proof) and the server
# spends ~50 ms verifying it before responding.
cf = client.counterfactual("bgv_zk_demo", x_denied, exp, bgv_zk=True)

print(f"\ncf_attestation_mode: {cf['cf_attestation_mode']}")

if cf["cf_attestation_mode"] == "ATTESTED_BGV_ZK":
    bgv_meta = cf.get("metadata", {}).get("bgv_zk", {})
    print("  ✓ The server cryptographically verified your lattice proof.")
    print(
        f"  Server-side verifier elapsed: "
        f"{bgv_meta.get('verifier_elapsed_ms')} ms"
    )
    print(f"  Encoding tag: {bgv_meta.get('encoding_tag')}")
    print(f"  Verdict: {bgv_meta.get('verdict')}")
elif cf["cf_attestation_mode"] == "UNATTESTED":
    bgv_meta = cf.get("metadata", {}).get("bgv_zk", {})
    if bgv_meta and bgv_meta.get("verdict", "").startswith("skipped"):
        print(
            "  Note: server received the lattice proof but CE_CF_USE_BGV_ZK is "
            "not set on this deployment. The proof was accepted but skipped; "
            "the response is on the v1-A path. The operator can flip the flag "
            "without any SDK changes."
        )
    else:
        print("  Server returned UNATTESTED — the v1-A path ran.")

# --- 6. Show the counterfactual (same shape as example 02) -------------
print(f"\ndecision_flipped: {cf['decision_flipped']}")
print("\nFeature changes (δ, sorted by |δ|):")
deltas = list(zip(feature_names, cf["delta"]))
for name, d in sorted(deltas, key=lambda kv: abs(kv[1]), reverse=True):
    if abs(d) > 1e-6:
        arrow = "↑" if d > 0 else "↓"
        print(f"  {name:>20}  {arrow} {abs(d):.4f}")

print("\nReason codes (Form C-1, ranked by |φ|):")
if cf["reason_codes"]:
    for c in cf["reason_codes"]:
        print(f"  [{c['form_c1_code']}] {c['form_c1_text']}")
        print(
            f"    feature={c['feature_name']}  "
            f"|φ|={c['phi_magnitude']:.4f}  "
            f"actionable={c['actionable']}"
        )


print(
    "\n--- bgv_zk threat-model recap ----------------------------------------\n"
    "ATTESTED_BGV_ZK ⇒ the server cannot have fabricated this SHAP output;\n"
    "                    the proof binds the response to the input the SDK\n"
    "                    submitted.\n"
    "ALSO       ⇒ the client (you) must submit the SAME δ_int in both\n"
    "                    ciphertexts. The proof does not enforce this; the\n"
    "                    SDK does (it generates both from the same vector).\n"
    "                    A malicious client could send inconsistent pairs;\n"
    "                    the protocol assumes you have no incentive to cheat\n"
    "                    against yourself (B2B internal attestation).\n"
    "FIT      ⇒ Banks attesting their own SHAP to a third-party auditor.\n"
    "             Healthcare providers attesting their own decisions.\n"
    "             Any case where the prover and the protected party align.\n"
    "NOT FIT  ⇒ Consumer-facing ECOA Reg-B / GDPR Art 22 adverse action\n"
    "             where the regulator is protecting the applicant FROM the\n"
    "             bank. For those, default bgv_zk=False (v1-A G1 Σ-IPA).\n"
)
