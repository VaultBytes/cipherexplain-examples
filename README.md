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
| [01_basic_lr.py](examples/01_basic_lr.py) | Train a logistic-regression locally, register the weights (no pickle / no training data), get encrypted SHAP attributions |
| [02_counterfactual.py](examples/02_counterfactual.py) | Counterfactual recourse + ECOA Reg-B Form C-1 reason codes from a single SHAP call. The customer's input never leaves their machine. |
| [03_feature_manifest.py](examples/03_feature_manifest.py) | Immutable + monotone-increase constraints (e.g. age can't change, income only goes up) |
| [04_tree_attested.py](examples/04_tree_attested.py) | RandomForest → server-side FHE leaf routing + local TreeExplainer; attestation auto-verified |
| [05_sklearn_pipeline.py](examples/05_sklearn_pipeline.py) | `Pipeline([scaler, classifier])` — the SDK strips the embedded scaler and uses it server-side |
| [06_async_batch.py](examples/06_async_batch.py) | Async webhook-delivered batch explanations for compliance workflows |

## What CipherExplain does

A drop-in SHAP API where the model is encrypted and your customer's data never leaves their environment. The server computes attributions on ciphertexts (CKKS FHE), and the SDK verifies the attestation chain (FreiKZG-SHAP soundness 2⁻²⁴⁹, vFHE Bulletproofs binding 2⁻¹²⁸, CRDC audit log, β₂ composition).

For credit denials and adverse-action notices, `client.counterfactual(...)` also returns:

- `x_prime` — the closest decision-flipping feature vector along the SHAP direction
- `reason_codes` — top-4 ECOA Regulation B Form C-1 codes ranked by `|φᵢ|` (excludes any features you mark immutable)

## Requirements

- Python 3.9+
- A CipherExplain API key — [free tier](https://vaultbytes.com/cipherexplain) covers 50 calls/month / 1 model

## Cryptography honesty

The v1-A counterfactual flow ships with `cf_attestation_mode = "UNATTESTED"`. The Pedersen homomorphism, real BLS12-381 G1 Σ-IPA π_CF, CRDC leaf, and β₂ composition are all real and verifiable. What's **not yet** algebraically bound is the link between `Enc(δ)` and `C_δ` — that's the Lyubashevsky lattice arm in v2. The forged-CF attack is closed structurally; replay across ciphertexts fails via Fiat-Shamir prefix binding.

See the [server release notes](https://github.com/VaultBytes/CipherExplain/releases/tag/sdk-v0.5.1) for the full picture.

## Contributing

Bug reports and example PRs welcome at [Issues](https://github.com/VaultBytes/cipherexplain-examples/issues). Email `b@vaultbytes.com` for commercial-license inquiries.

## License

AGPL-3.0-or-later. Commercial licences available on request.
