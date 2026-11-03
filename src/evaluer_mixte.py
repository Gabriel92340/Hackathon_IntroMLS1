import sys
import numpy as np
from sklearn.metrics import f1_score
from evaluer import plis, y, u_tr, u_te, total, connus_de
from quotas import affecter_cout

clfs = sys.argv[1].split("+")
regs = sys.argv[2].split("+")
for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
    oof = np.zeros(len(y), dtype=int)
    for f, (a, b) in enumerate(plis):
        p = sum(np.vstack([np.load(f"preds/{c}_f{f}_val.npy"), np.load(f"preds/{c}_f{f}_test.npy")]) for c in clfs) / len(clfs)
        s = sum(np.concatenate([np.load(f"preds/{r}_f{f}_val.npy"), np.load(f"preds/{r}_f{f}_test.npy")]) for r in regs) / len(regs)
        cout = -np.log(np.clip(p, 1e-9, 1)) + lam * (np.arange(3)[None, :] - s[:, None]) ** 2
        pred = affecter_cout(cout, np.concatenate([u_tr[b], u_te]), total, connus_de(a))
        oof[b] = pred[:len(b)]
    print(f"{sys.argv[1]} | {sys.argv[2]} | lambda {lam}: {f1_score(y, oof, average='macro'):.4f}", flush=True)
