# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""06 — XGBoost and LightGBM gradient boosters.

Both use ``register_xgboost`` / ``register_lightgbm`` helpers that
extract the JSON booster dump and ship it as a tree-ensemble spec.
``explain_tree`` then runs FHE leaf routing on the server with local
TreeExplainer verification on the dump.

Optional install:
  pip install 'cipherexplain[fhe]' xgboost lightgbm

Run:
  CIPHEREXPLAIN_API_KEY=vb_... python examples/06_xgboost_lightgbm.py
"""
import os
import sys

import numpy as np
from sklearn.datasets import make_classification

from cipherexplain_sdk import CipherExplainClient

API_KEY = os.environ.get("CIPHEREXPLAIN_API_KEY") or sys.exit(
    "Set CIPHEREXPLAIN_API_KEY")
client = CipherExplainClient(api_key=API_KEY)

X, y = make_classification(n_samples=400, n_features=6, n_informative=4,
                            random_state=3)
feature_names = [f"f{i}" for i in range(6)]

# --- XGBoost ---------------------------------------------------------------
try:
    import xgboost as xgb
except ImportError:
    print("skip XGBoost: pip install xgboost")
else:
    booster = xgb.XGBClassifier(
        n_estimators=20, max_depth=4, use_label_encoder=False,
        eval_metric="logloss",
    ).fit(X, y)
    try:
        client.delete("xgb_demo")
    except Exception:
        pass
    client.register_xgboost("xgb_demo", booster, feature_names=feature_names)
    r = client.explain_tree("xgb_demo", X[0].tolist(), model=booster)
    ta = r["tree_attested"]
    print(f"XGBoost  pred={ta.prediction:.4f}  attestation={ta.attestation_verified}")
    print(f"         top-3 |φ|: " + ", ".join(
        f"{n}={p:+.3f}" for n, p in sorted(
            zip(feature_names, ta.shap_values),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:3]
    ))
    client.delete("xgb_demo")

# --- LightGBM --------------------------------------------------------------
try:
    import lightgbm as lgb
except ImportError:
    print("skip LightGBM: pip install lightgbm")
else:
    gbm = lgb.LGBMClassifier(n_estimators=20, max_depth=4,
                              verbose=-1).fit(X, y)
    try:
        client.delete("lgb_demo")
    except Exception:
        pass
    client.register_lightgbm("lgb_demo", gbm, feature_names=feature_names)
    r = client.explain_tree("lgb_demo", X[0].tolist(), model=gbm)
    ta = r["tree_attested"]
    print(f"LightGBM pred={ta.prediction:.4f}  attestation={ta.attestation_verified}")
    print(f"         top-3 |φ|: " + ", ".join(
        f"{n}={p:+.3f}" for n, p in sorted(
            zip(feature_names, ta.shap_values),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:3]
    ))
    client.delete("lgb_demo")
