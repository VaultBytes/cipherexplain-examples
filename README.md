<!--
Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
Patent pending: PCT/IB2026/053378, PCT/IB2026/053405
-->

# CipherExplain examples

Runnable examples for the [CipherExplain](https://vaultbytes.com/cipherexplain) Python SDK. Each script in [`examples/`](examples/) is self-contained: clone, install, drop in your API key, run.

[![PyPI](https://img.shields.io/pypi/v/cipherexplain.svg)](https://pypi.org/project/cipherexplain/)
[![Python](https://img.shields.io/pypi/pyversions/cipherexplain.svg)](https://pypi.org/project/cipherexplain/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## What is CipherExplain?

**Encrypted explainable AI.** CipherExplain computes SHAP feature attributions for binary classifiers under fully homomorphic encryption (CKKS). The customer's input is never decrypted on the server; the server returns attributions that the SDK decrypts locally.

It's built for the regulated-ML stack:

- **Banks and fintechs** — under US ECOA Regulation B and EU GDPR Article 22 you must give a denied applicant both an explanation and a counterfactual ("what would have to change for approval"). Vanilla SHAP exposes the applicant's data to whoever runs the inference; CipherExplain doesn't.
- **Healthcare and insurance** — clinical-decision explanations under HIPAA / GDPR Article 9 where the patient's record cannot leave the controlling environment.
- **Any audit-grade ML deployment** — the response carries an integrity stack the auditor can replay off-line.

The headline v0.5.1 feature is **counterfactual recourse with ECOA Form C-1 reason codes**: the customer asks "why was I denied?" and gets back the closest decision-flipping change to their feature vector plus the four regulatory reason codes ranked by SHAP magnitude. Protected-class features (age, sex, ...) are cryptographically scrubbed — they can't appear in the reason codes or the recourse.

Under the hood:

- **SDK** (your code, your machine): trains your model locally, ships only weights to the API. Optionally encrypts inputs under your own CKKS key via `fhe_mode='ckks'`.
- **Server** ([cipherexplain.vaultbytes.com](https://cipherexplain.vaultbytes.com)): evaluates the model and computes SHAP on the ciphertext, returns encrypted SHAP values + attestation.
- **You decrypt locally** with your own key and verify attestation via `verify_cluster_a_response()`.

Drop-in for `shap.LinearExplainer` / `shap.TreeExplainer` workflows; wire protocol is JSON over HTTPS.

## Quick start

```bash
git clone https://github.com/VaultBytes/cipherexplain-examples.git
cd cipherexplain-examples

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export CIPHEREXPLAIN_API_KEY="vb_..."   # get one at vaultbytes.com/cipherexplain
python examples/01_basic_lr.py
```

## What's inside

| Example | What it shows |
|---|---|
| [01_basic_lr.py](examples/01_basic_lr.py) | Train an LR locally, register the weights (no pickle / no training data), get encrypted SHAP attributions |
| [02_counterfactual.py](examples/02_counterfactual.py) | Counterfactual recourse + ECOA Reg-B Form C-1 reason codes |
| [03_feature_manifest.py](examples/03_feature_manifest.py) | Immutable + monotone constraints (age stays fixed, income only goes up, ...) |
| [04_tree_attested.py](examples/04_tree_attested.py) | RandomForest → server FHE leaf routing + local TreeExplainer with auto-verified attestation |
| [05_sklearn_pipeline.py](examples/05_sklearn_pipeline.py) | `Pipeline([scaler, classifier])` — scaler params shipped, applied server-side |
| [06_xgboost_lightgbm.py](examples/06_xgboost_lightgbm.py) | XGBoost + LightGBM gradient boosters via `register_xgboost` / `register_lightgbm` |
| [07_mlp_ckks.py](examples/07_mlp_ckks.py) | Multi-layer perceptron under full CKKS evaluation |
| [09_dp_shap.py](examples/09_dp_shap.py) | (ε, δ)-differential-privacy noise on published attributions + budget tracking |
| [10_cluster_a_verify.py](examples/10_cluster_a_verify.py) | Cluster-A attestation: re-derive composition β + CRDC leaf locally |
| [12_local_fhe_mode.py](examples/12_local_fhe_mode.py) | Client-side CKKS encryption — server never sees plaintext input |

> Async-batch (`/explain/batch` webhook delivery) and API-key rotation are operational utilities documented in the [SDK reference](https://vaultbytes.com/cipherexplain) — not feature demos.

## Requirements

- Python 3.9+
- A CipherExplain API key — the [free tier](https://vaultbytes.com/cipherexplain) covers 50 calls/month / 1 model

Optional per-example:

- `xgboost` / `lightgbm` for example 06
- `cipherexplain[fhe]` (adds `openfhe`) for example 12

## Honest about what's attested

The counterfactual response carries a flag, `cf_attestation_mode`. In v0.5.1 it always reads `"UNATTESTED"`. Here's what that means in plain language.

### What we promise today

Three things you can rely on right now:

1. **The customer cannot forge the "fixed" applicant.** The API is designed so the customer never submits the encrypted post-counterfactual input. The server constructs it from the customer's original encrypted input plus the small change they submit. There is no field on the request where a malicious customer could inject a fabricated ciphertext claiming "this is the version of me the model approves."

2. **Commitments are real and tamper-evident.** When the SDK builds the request, it commits cryptographically to both the original input and the proposed change. The server checks these commitments add up correctly. If anyone — customer or attacker — tampered with either commitment in transit, the check fails and the server returns an error.

3. **The proof of "I know what I committed to" is real.** Along with the request, the SDK ships a zero-knowledge proof showing it actually knows the values it committed to (not just random bytes that happen to look like commitments). This is a real elliptic-curve Sigma-protocol on BLS12-381, the same curve Ethereum 2.0 and Zcash use. Replaying the proof against a different encrypted input fails, because the proof is bound to the specific ciphertext via a standard Fiat-Shamir transform.

4. **Everything lands in an audit log.** Each counterfactual call appends a Merkle-tree leaf to a per-tenant CRDC log on the server. An auditor can ask the server for the tree root weeks later and reconstruct exactly which counterfactuals were issued for which inputs, in what order. The response also carries a composition tag (`β₂`) that ties this call to the underlying model and the customer's original `/explain` call — the auditor re-derives `β₂` from public response material and checks the bytes match.

### What we don't promise yet

There is one specific cryptographic link we have **not** built into v0.5.1: the algebraic bond between the **encrypted** small change and the **committed** small change. They live in different mathematical objects (the encryption is a CKKS lattice ciphertext; the commitment is a point on an elliptic curve), and proving they're consistent in zero-knowledge over CKKS's approximate arithmetic is an open research area (see Li et al., IACR 2025/382). That bond ships in a later release.

What does this gap let a sophisticated attacker do, and what doesn't it?

- It does **not** let a customer get a counterfactual based on data the server didn't see. The server-side input is fixed at `/explain` time and the customer can't change it.
- It does **not** let an attacker forge the audit log or the reason codes — those run off the server's computation, not the customer's claim.
- A malicious **client** could in principle send a commitment to one small change while encrypting a different one. The result: the audit log records one story, the customer decrypts another. The client is the only party who can detect this, by decrypting and checking. Under the standard "honest-but-curious server, self-interested client" threat model that regulated ML deployments operate under, this is not exploitable — the client gains nothing by lying to themselves.

That's the entire gap. Until the lattice arm lands, `cf_attestation_mode` will read `"UNATTESTED"` to keep the record clean. The day it does, the same code paths flip to `"ATTESTED"` with no API change.

## Contributing

Bug reports and example PRs welcome on [Issues](https://github.com/VaultBytes/cipherexplain-examples/issues). Email `b@vaultbytes.com` for commercial-license inquiries.

## License

AGPL-3.0-or-later. Commercial licences available on request.
