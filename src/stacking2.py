import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de
from quotas import affecter_cout
from commun import charger, CLASSES

CLF = ["hgb_clf_pr", "et_clf", "et_clf_r", "et_clf_r_f1"]
REG = ["et_reg"]
mode = sys.argv[1] if len(sys.argv) > 1 else "lr_calib"
sortie = sys.argv[2] if len(sys.argv) > 2 else None
n, nt = len(y), len(u_te)


def charge(f, partie):
    P = [np.load(f"preds/{c}_f{f}_{partie}.npy") for c in CLF]
    R = [np.load(f"preds/{r}_f{f}_{partie}.npy") for r in REG]
    return P, R


def base_feats(P, R):
    cols = [np.log(np.clip(p, 1e-6, 1)) for p in P]
    for s in R:
        cols.append(np.column_stack([s, s ** 2]))
    return np.hstack(cols)


def score(P, R):
    e = sum(p[:, 1] + 2 * p[:, 2] for p in P) / len(P)
    return 0.5 * e + 0.5 * sum(R) / len(R)


def calib(s_req, g_req, s_ref, g_ref, y_ref):
    out = np.full((len(s_req), 4), np.nan)
    df = pd.DataFrame({"g": g_ref, "s": s_ref, "y": y_ref})
    groupes = {g: d for g, d in df.groupby("g")}
    for g in np.unique(g_req):
        iq = np.where(g_req == g)[0]
        if g not in groupes:
            continue
        d = groupes[g]
        cols = []
        for c in range(3):
            sc = np.sort(d["s"].values[d["y"].values == c])
            cols.append(np.searchsorted(sc, s_req[iq]) / max(len(sc), 1) if len(sc) else np.full(len(iq), 0.5))
        tout_s = np.sort(d["s"].values)
        cols.append(np.searchsorted(tout_s, s_req[iq]) / len(tout_s))
        out[iq] = np.column_stack(cols)
    return np.nan_to_num(out, nan=0.5)


s_oof = np.zeros(n)
B_oof = None
for f, (a, b) in enumerate(plis):
    P, R = charge(f, "val")
    s_oof[b] = score(P, R)
    bf = base_feats(P, R)
    if B_oof is None:
        B_oof = np.zeros((n, bf.shape[1]))
    B_oof[b] = bf

C_oof = np.zeros((n, 4))
for f, (a, b) in enumerate(plis):
    C_oof[b] = calib(s_oof[b], u_tr[b], s_oof[a], u_tr[a], y[a])


def meta_X(B, C):
    return np.hstack([B, C]) if "calib" in mode else B


def fab():
    if mode.startswith("hgb"):
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    return LogisticRegression(C=1.0, max_iter=3000)


M_oof = meta_X(B_oof, C_oof)
pred_oof = np.zeros(n, dtype=int)
p_test_tot = np.zeros((nt, 3))
for f, (a, b) in enumerate(plis):
    m = fab().fit(M_oof[a], y[a])
    Pt, Rt = charge(f, "test")
    Ct = calib(score(Pt, Rt), u_te, s_oof[a], u_tr[a], y[a])
    pv = m.predict_proba(M_oof[b])
    pt = m.predict_proba(meta_X(base_feats(Pt, Rt), Ct))
    pred = affecter_cout(-np.log(np.clip(np.vstack([pv, pt]), 1e-9, 1)), np.concatenate([u_tr[b], u_te]), total, connus_de(a))
    pred_oof[b] = pred[:len(b)]

scores = [f1_score(y[b], pred_oof[b], average="macro") for a, b in plis]
print(f"{mode}: {f1_score(y, pred_oof, average='macro'):.4f}  folds {np.round(scores, 4)}", flush=True)

if sortie:
    m = fab().fit(M_oof, y)
    p_test = np.zeros((nt, 3))
    for f, (a, b) in enumerate(plis):
        Pt, Rt = charge(f, "test")
        Ct = calib(score(Pt, Rt), u_te, s_oof, u_tr, y)
        p_test += m.predict_proba(meta_X(base_feats(Pt, Rt), Ct)) / 5
    train, test = charger()
    pred = affecter_cout(-np.log(np.clip(p_test, 1e-9, 1)), u_te, total, connus_de(np.arange(n)))
    pd.DataFrame({"workout_id": test["workout_id"], "intensity": [CLASSES[i] for i in pred]}).to_csv(sortie, index=False)
    print("écrit", sortie)
