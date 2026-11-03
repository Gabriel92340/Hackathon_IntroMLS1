import numpy as np
from scipy.optimize import linear_sum_assignment


CONVENTION = "hard"


def comptes_tertiles(total):
    k, r = divmod(int(total), 3)
    if CONVENTION == "hard":
        return np.array([k, k, k]) + np.array({0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 1]}[r])
    return np.full(3, total / 3)


DELTA = 0.0


def quotas_entiers(n_inconnus, total, comptes_connus):
    cible = comptes_tertiles(total) - comptes_connus
    cible = np.clip(cible, 0, None).astype(float)
    cible = cible * n_inconnus / cible.sum() if cible.sum() > 0 else np.full(3, n_inconnus / 3)
    if DELTA:
        cible = np.clip(cible + np.array([-0.5, 1.0, -0.5]) * DELTA * n_inconnus, 0, None)
        cible = cible * n_inconnus / cible.sum()
    base = np.floor(cible).astype(int)
    reste = n_inconnus - base.sum()
    ordre = np.argsort(-(cible - base))
    base[ordre[:reste]] += 1
    return base


def affecter(proba, users, total_par_user, connus_par_user):
    pred = proba.argmax(1).copy()
    logp = np.log(np.clip(proba, 1e-9, 1))
    for u in np.unique(users):
        idx = np.where(users == u)[0]
        q = quotas_entiers(len(idx), total_par_user[u], connus_par_user[u])
        colonnes = np.repeat(np.arange(3), q)
        cout = -logp[idx][:, colonnes]
        lignes, cols = linear_sum_assignment(cout)
        pred[idx[lignes]] = colonnes[cols]
    return pred


def affecter_cout(cout, users, total_par_user, connus_par_user):
    pred = cout.argmin(1).copy()
    for u in np.unique(users):
        idx = np.where(users == u)[0]
        q = quotas_entiers(len(idx), total_par_user[u], connus_par_user[u])
        colonnes = np.repeat(np.arange(3), q)
        lignes, cols = linear_sum_assignment(cout[idx][:, colonnes])
        pred[idx[lignes]] = colonnes[cols]
    return pred
