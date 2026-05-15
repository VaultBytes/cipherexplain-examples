# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""10 — Cluster-A attestation verification (CRDC + composition β).

The Cluster-A integrity stack lets the client re-derive the server's
``composition β`` from public response material and check it matches
the published bytes — without trusting the server. Verifies:

  * β = HKDF(model_commit || input_ct || tenant_root || ts || circuit)
  * CRDC leaf reconstruction from announced (nullifier, C_phi, C_x)
  * Per-domain labels (EDAW, FCRAS-B, CRDC)

Server side must run with CE_CRDC_ENABLED=1 and CE_STACK_COMPOSITION=1
(both on by default on cipherexplain.vaultbytes.com).

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/09_cluster_a_verify.py
"""
import os
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cipherexplain_sdk import (
    CipherExplainClient, ClusterAVerificationResult, extract_spec,
    verify_cluster_a_response,
)

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=200, n_features=5, random_state=7)
feature_names = [f"f{i}" for i in range(5)]
model = LogisticRegression(max_iter=1000).fit(X, y)
spec = extract_spec(model, "cluster_a_demo", feature_names=feature_names)

try:
    client.delete("cluster_a_demo")
except Exception:
    pass
client.register(spec)

# Explain — response carries metadata.composition (β + domain labels) +
# metadata.crdc (Merkle leaf material) when CE_CRDC_ENABLED=1 +
# CE_STACK_COMPOSITION=1 on the server.
result = client.explain_raw("cluster_a_demo", X[0].tolist())

# Verify both components.
verdict: ClusterAVerificationResult = verify_cluster_a_response(
    result, x=X[0].tolist(),
)

print(f"composition β verdict: {verdict.composition.status}")
print(f"  derived β = {verdict.composition.derived_beta_hex[:16]}...")
print(f"  published = {verdict.composition.published_beta_hex[:16]}...")
print(f"  match     = {verdict.composition.match}")

print(f"\nCRDC verdict: {verdict.crdc.status}")
print(f"  reproducible_leaf = {verdict.crdc.match if hasattr(verdict.crdc, 'match') else 'n/a'}")

print(f"\noverall: {verdict.overall_status}")

# CRDC root inclusion proof (audit-trail).
# The server publishes the current tenant root via GET /audit/crdc/root.
# An auditor can fetch (root, leaf, path) and run crdc.verify_inclusion
# off-line. For SDK convenience the verdict surfaces the leaf-side bytes;
# fetching the root + Merkle path is a follow-up tool call.

client.delete("cluster_a_demo")
