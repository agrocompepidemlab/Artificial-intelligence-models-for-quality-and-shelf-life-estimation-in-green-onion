"""Fase 3 v2 · datos, modelos y un pliegue de validación (lo importa cada proceso del runner).

Modelos (hiperparámetros fijados en PLAN_ANALISIS_v2.md):
  M1 CTD-GB v2  descriptores sin escala → HistGradientBoosting
  M2 CNN-ridge  embedding ResNet18 del recorte → Ridge / regresión logística (regularización por CV interna agrupada)
  Ens           media de M1 y M2 (probabilidades en las clases)
Variantes «+DAH»: el DAH como entrada adicional. Referencias: trivial, solo DAH, solo genotipo, genotipo + DAH.
Por pliegue: (a) todo el entrenamiento → predicciones puntuales; (b) 85 % de las unidades → calibración conformal
(α = 0,10) en el 15 % restante (la misma unidad que el protocolo deja fuera)."""
import os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))         # el repositorio
DATA = os.environ.get("GREEN_ONION_DATA", os.path.join(_ROOT, "data"))
CACHE = os.environ.get("GREEN_ONION_V2", os.path.join(DATA, "cache_v2"))
LAB_FILE = os.path.join(_ROOT, "data", "Phase_2_Physicochemical.xlsx")
ALPHA = 0.10
M1 = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, min_samples_leaf=40)
RIDGE_ALPHAS = np.logspace(-2, 5, 15)
LOGIT_C = np.logspace(-5, 0, 6)
SL_SURV = {"CUNSBR048": 12.80, "91400070": 10.14, "ANTSNC003": 15.35, "91400028": 12.31, "NDSCAR038": 5.96,
           "91400035": 5.37, "ANTSON013": 6.29, "ANTSRO029": 14.14, "91400071": 8.38, "ANTSRO030": 11.29}
LOT_A = {"91400028", "91400035", "ANTSON013", "CUNSBR048", "NDSCAR038"}

# ---------------------------------------------------------------- datos
df = pd.read_csv(os.path.join(DATA, "dataset_final.csv"), usecols=["ruta", "tiempo", "genotipo", "dah", "calidad_idx"])
df["geno"] = df["genotipo"].astype(str).replace({"CUNSBRO048": "CUNSBR048"})
lab = pd.read_excel(LAB_FILE)
_V = ["Pseudosteam length (cm)", "Pseudosteam Diameter (cm)", "Pseudosteam Hue", "Leave Hue", "Firmness (N)"]
_W = np.array([74.5, 72.5, 75, 73, 84]); _W = _W / _W.sum()
_m = lab[lab["Experimento"] == 2].groupby(["Genotipo", "Tiempo"])[_V].mean()
_u = np.column_stack([np.clip((_m[_V[0]] - 15) / 15, 0, 1), np.clip((_m[_V[1]] - 5) / 18, 0, 1),
                      np.exp(-(_m[_V[2]] - 102.9) ** 2 / (2 * 16 ** 2)),
                      np.exp(-np.minimum(0, _m[_V[3]] - 124.1) ** 2 / (2 * 5.2 ** 2)), np.clip(1 - (_m[_V[4]] - 3) / 27, 0, 1)])
_g2 = pd.Series(100 * (_u * _W).sum(1), index=_m.index)
df["gqi_e2"] = [_g2[(g, t)] for g, t in zip(df["geno"], df["tiempo"])]
df["sl"] = df["geno"].map(SL_SURV).clip(upper=14.0)
df["rsl"] = df["sl"] - df["dah"]
df["block"] = pd.factorize(df["geno"] + "_" + df["tiempo"].astype(str))[0]
df["lot"] = np.where(df["geno"].isin(LOT_A), "A", "B")
df["day"] = pd.factorize(df["lot"] + df["tiempo"].astype(str))[0]
df["session"] = pd.factorize(df["ruta"].str.rsplit("\\", n=1).str[0])[0]
_tr = pd.read_csv(os.path.join(DATA, "cache", "tramos.csv"))
df["pseudo"] = pd.factorize(df["session"].astype(str) + "_" + df["ruta"].map(_tr.set_index("ruta")["tramo"]).astype(str))[0]

_z = np.load(os.path.join(CACHE, "features_v2.npz"), allow_pickle=True)
NAMES = list(_z["names"])
MODEL_COLS = [i for i, n in enumerate(NAMES) if not n.startswith(("raw_", "cor_"))]
X1 = _z["D"][:, MODEL_COLS].astype(np.float32)
X2 = _z["E"].astype(np.float32)
meta = pd.read_csv(os.path.join(CACHE, "meta_v2.csv"))
assert (meta["ruta"].to_numpy() == df["ruta"].to_numpy()).all()
OKP = meta["ok"].to_numpy(bool)
REG = ["dah", "rsl", "gqi_e2"]
Y = {t: df[t].to_numpy(np.float64) for t in REG}
C = df["calidad_idx"].to_numpy(int)
DAH = df["dah"].to_numpy(float)
SW = 1.0 / df.groupby("session")["ruta"].transform("size").to_numpy(float)   # cada sesión pesa lo mismo
N = len(df)
INNER_UNIT = {"P1": "block", "P1_day": "day", "P2": "block", "P3": "geno", "P4_lot": "geno",
              "photo": "session", "pseudo": "session", "session": "session"}


def outer_splits(scheme, seed=42):
    n = np.arange(N)
    if scheme in ("P1", "photo", "pseudo", "session"):
        col = {"P1": df["block"].to_numpy(), "photo": np.arange(N), "pseudo": df["pseudo"].to_numpy(),
               "session": df["session"].to_numpy()}[scheme]
        u_ = np.unique(col); rng = np.random.default_rng(seed); rng.shuffle(u_)
        f = pd.Series(col).map({v: i % 5 for i, v in enumerate(u_)}).to_numpy()
        return [(n[f != k], n[f == k]) for k in range(5)]
    g = {"P1_day": df["day"], "P2": df["tiempo"], "P3": df["geno"], "P4_lot": df["lot"]}[scheme].to_numpy()
    return list(LeaveOneGroupOut().split(n, groups=g))


def calib_split(train, scheme, seed):
    b = df[INNER_UNIT[scheme]].to_numpy()[train]; u_ = np.unique(b); rng = np.random.default_rng(seed); rng.shuffle(u_)
    k = max(1, round(0.15 * len(u_)))
    cal = np.isin(b, u_[:k])
    return train[~cal], train[cal]


def conformal_q(res):
    res = res[np.isfinite(res)]
    n = len(res); lev = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(np.abs(res), lev, method="higher"))


# ---------------------------------------------------------------- modelos
class Std:
    def __init__(self, X, w):
        sw = w / w.sum(); self.m = sw @ X; self.s = np.sqrt(sw @ (X - self.m) ** 2); self.s[self.s == 0] = 1

    def __call__(self, X):
        return (X - self.m) / self.s


def ridge_path(Z, y, w, alphas):
    """Ridge ponderado para todos los α con una sola descomposición: min Σ w (y − b − zβ)² + α‖β‖²."""
    sw = w / w.sum(); zm = sw @ Z; ym = sw @ y
    Zc = Z - zm
    G = Zc.T @ (Zc * w[:, None]); bvec = Zc.T @ (w * (y - ym))
    lam, V = np.linalg.eigh(G.astype(np.float64)); Vb = V.T @ bvec
    return zm, ym, [V @ (Vb / (lam + a)) for a in alphas]


def fit_m1(i, X, y, is_class):
    i = i[OKP[i]]
    if is_class:
        return HistGradientBoostingClassifier(**M1, random_state=42).fit(X[i], C[i], sample_weight=SW[i])
    return HistGradientBoostingRegressor(**M1, random_state=42).fit(X[i], y[i], sample_weight=SW[i])


def fit_m2(i, X, y, is_class, scheme):
    i = i[OKP[i]]
    std = Std(X[i], SW[i]); Z = std(X[i]); g = df[INNER_UNIT[scheme]].to_numpy()[i]
    k = min(4, len(np.unique(g)))
    grid = LOGIT_C if is_class else RIDGE_ALPHAS
    score = np.zeros(len(grid))
    for a, b in GroupKFold(k).split(Z, groups=g):
        if is_class:
            for j, h in enumerate(grid):
                mdl = LogisticRegression(C=h, max_iter=300).fit(Z[a], C[i][a], sample_weight=SW[i][a] / SW[i][a].mean())
                pr = np.clip(mdl.predict_proba(Z[b]), 1e-9, 1)
                score[j] += -np.sum(SW[i][b] * np.log(pr[np.arange(len(b)), C[i][b]]))
        else:
            zm, ym, betas = ridge_path(Z[a], y[i][a], SW[i][a], grid)
            for j, beta in enumerate(betas):
                score[j] += np.sum(SW[i][b] * (y[i][b] - ((Z[b] - zm) @ beta + ym)) ** 2)
    best = grid[int(np.argmin(score))]
    if is_class:
        mdl = LogisticRegression(C=best, max_iter=500).fit(Z, C[i], sample_weight=SW[i] / SW[i].mean())
        return (lambda Xn: mdl.predict_proba(std(Xn))), best
    zm, ym, (beta,) = ridge_path(Z, y[i], SW[i], [best])
    return (lambda Xn: (std(Xn) - zm) @ beta + ym), best


def fit_set(i, scheme, targets, dah_input=False):
    X1_, X2_ = (np.column_stack([X1, DAH]), np.column_stack([X2, DAH])) if dah_input else (X1, X2)
    models, alphas = {}, {}
    for t in targets:
        is_c = t == "class"; y = None if is_c else Y[t]
        m2, a = fit_m2(i, X2_, y, is_c, scheme)
        models[t] = (fit_m1(i, X1_, y, is_c), m2); alphas[t] = a
    return models, alphas, X1_, X2_


def predict_all(models, X1_, X2_, idx, prefix=""):
    out = {}
    for t, (m1, m2) in models.items():
        if t == "class":
            p1 = m1.predict_proba(X1_[idx]); p2 = m2(X2_[idx])
        else:
            p1 = m1.predict(X1_[idx]); p2 = m2(X2_[idx])
        for name, v in [("M1", p1), ("M2", p2), ("Ens", (p1 + p2) / 2)]:
            v = np.asarray(v, float).copy(); v[~OKP[idx]] = np.nan
            out[f"{name}{prefix}|{t}"] = v
    return out


def baselines(tr, te, y):
    g = df["geno"].to_numpy()
    out = {"Trivial": np.full(len(te), np.average(y[tr], weights=SW[tr]))}
    out["DAH only"] = LinearRegression().fit(DAH[tr, None], y[tr], sample_weight=SW[tr]).predict(DAH[te, None])
    mg = pd.Series(y[tr]).groupby(g[tr]).mean()
    out["Genotype only"] = pd.Series(g[te]).map(mg).fillna(y[tr].mean()).to_numpy()
    G = pd.get_dummies(pd.Categorical(g, categories=sorted(set(g))), dtype=float).to_numpy()
    Zm = np.column_stack([DAH, G])
    p = LinearRegression().fit(Zm[tr], y[tr], sample_weight=SW[tr]).predict(Zm[te])
    unseen = ~np.isin(g[te], np.unique(g[tr]))
    p[unseen] = out["DAH only"][unseen]
    out["Genotype + DAH"] = p
    return out


def class_baselines(tr, te):
    out = {"Trivial": np.tile(np.bincount(C[tr], minlength=4) / len(tr), (len(te), 1))}
    for name, col in [("DAH only", df["tiempo"].to_numpy()), ("Genotype only", df["geno"].to_numpy())]:
        tab = pd.crosstab(col[tr], C[tr]).reindex(columns=range(4), fill_value=0)
        tab = (tab + 1).div((tab + 1).sum(1), axis=0)
        out[name] = pd.DataFrame(index=col[te]).join(tab).fillna(0.25).to_numpy()
    return out


def run_fold(scheme, seed, k, ladder=False):
    """Un pliegue: devuelve los índices de prueba, las predicciones para ellos y los radios conformales."""
    train, test = outer_splits(scheme, seed)[k]
    P, q = {}, {"fold": k}
    models, alphas, X1_, X2_ = fit_set(train, scheme, REG + ["class"])
    P.update(predict_all(models, X1_, X2_, test))
    q.update({f"alpha|{t}": a for t, a in alphas.items()})
    if not ladder:
        md, _, X1d, X2d = fit_set(train, scheme, ["rsl", "gqi_e2", "class"], dah_input=True)
        P.update(predict_all(md, X1d, X2d, test, prefix="+DAH"))
        fit, cal = calib_split(train, scheme, seed + k)
        mc, _, X1c, X2c = fit_set(fit, scheme, REG)
        pc_cal, pc_te = predict_all(mc, X1c, X2c, cal), predict_all(mc, X1c, X2c, test)
        for kk in pc_te:
            t = kk.split("|")[1]
            q[kk] = conformal_q(Y[t][cal] - pc_cal[kk])
            P["cc_" + kk] = pc_te[kk]
        for t in REG:
            for name, v in baselines(fit, cal, Y[t]).items():
                q[f"{name}|{t}"] = conformal_q(Y[t][cal] - v)
            for name, v in baselines(fit, test, Y[t]).items():
                P[f"cc_{name}|{t}"] = v
    for t in REG:
        for name, v in baselines(train, test, Y[t]).items():
            P[f"{name}|{t}"] = v
    for name, v in class_baselines(train, test).items():
        P[f"{name}|class"] = v
    return test, P, q
