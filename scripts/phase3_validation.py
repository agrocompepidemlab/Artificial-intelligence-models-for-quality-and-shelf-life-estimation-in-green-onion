"""Fase 3 — validación del reencuadre con los DOS modelos (CTD-GB y CNN-MTL), referencias y conformal.

Esquemas (unidad que queda fuera completa en cada pliegue):
  P1        sesión = bloque genotipo × tiempo (las 4 carpetas fotografiadas seguidas), 5 pliegues
  P1_day    día de captura (8 grupos; 5 genotipos por día), sensibilidad de P1
  P2        tiempo de almacenamiento no visto (4 pliegues)
  P3        genotipo no visto (10 pliegues)
En cada pliegue los grupos de entrenamiento se dividen en entrenamiento (70 %), validación (15 %, parada
temprana del CNN-MTL) y calibración conformal (15 %). Intervalos conformales divididos (split conformal,
Angelopoulos y Bates, arXiv:2107.07511): q = cuantil ceil((n+1)(1-α))/n de |y - ŷ| en calibración, α = 0,10.
Salidas: dah, rsl (vida útil restante = t50 en línea de la Fase 1, tope 14 d, menos el DAH), gqi_e2 (GQI del
experimento 2, el de las fotos) y la clase visual (4 clases)."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]                       # el repositorio
DATA = Path(os.environ.get("GREEN_ONION_DATA", ROOT / "data" / "phase3"))   # dataset_final.csv y cachés de rasgos
PHOTOS = Path(os.environ.get("GREEN_ONION_PHOTOS", DATA / "photos"))   # carpeta que contiene T1–T4
LAB_FILE = ROOT / "data" / "Phase_2_Physicochemical.xlsx"


def photo_path(ruta):
    """La foto en este equipo: la ruta guardada en dataset_final.csv se reubica en PHOTOS."""
    parts = Path(str(ruta).replace(chr(92), "/")).parts
    return PHOTOS.joinpath(*parts[parts.index("GreenOnionCap3") + 1:])

import sys, time, pickle
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

OUT = sys.argv[1]
SCHEMES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["P1", "P2", "P3", "P1_day"]
SEED, ALPHA = 42, 0.10
import os
QUICK = os.environ.get("QUICK") == "1"     # prueba: 1 pliegue, 2 épocas
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
CTD_GB = dict(max_iter=400, learning_rate=0.05, max_depth=7, l2_regularization=1.0)
CNN = dict(hidden=160, dropout=(0.40, 0.30), input_noise=0.05, learning_rate=5e-4, weight_decay=1e-3,
           batch_size=256, max_epochs=60, patience=10, scheduler_factor=0.5, scheduler_patience=4,
           label_smoothing=0.05, auxiliary_weight=0.3)
SL_SURV = {"CUNSBR048": 12.80, "91400070": 10.14, "ANTSNC003": 15.35, "91400028": 12.31, "NDSCAR038": 5.96,
           "91400035": 5.37, "ANTSON013": 6.29, "ANTSRO029": 14.14, "91400071": 8.38, "ANTSRO030": 11.29}

# ---------------------------------------------------------------- datos
df = pd.read_csv(DATA / "dataset_final.csv")
X = np.load(DATA / "cache_descriptores_mejorados.npz", allow_pickle=True)["X"].astype(np.float32)
X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
emb = np.load(DATA / "cache_embeddings_cnn.npz")["X"].astype(np.float32)
XB = np.hstack([X, emb])
df["geno"] = df["genotipo"].astype(str).replace({"CUNSBRO048": "CUNSBR048"})
lab = pd.read_excel(LAB_FILE)
V = ["Pseudosteam length (cm)", "Pseudosteam Diameter (cm)", "Pseudosteam Hue", "Leave Hue", "Firmness (N)"]
W = np.array([74.5, 72.5, 75, 73, 84]); W = W / W.sum()
m = lab[lab["Experimento"] == 2].groupby(["Genotipo", "Tiempo"])[V].mean()
u = np.column_stack([np.clip((m[V[0]] - 15) / 15, 0, 1), np.clip((m[V[1]] - 5) / 18, 0, 1),
                     np.exp(-(m[V[2]] - 102.9) ** 2 / (2 * 16 ** 2)),
                     np.exp(-np.minimum(0, m[V[3]] - 124.1) ** 2 / (2 * 5.2 ** 2)), np.clip(1 - (m[V[4]] - 3) / 27, 0, 1)])
g2 = pd.Series(100 * (u * W).sum(1), index=m.index)
df["gqi_e2"] = [g2[(g, t)] for g, t in zip(df["geno"], df["tiempo"])]
df["sl"] = df["geno"].map(SL_SURV).clip(upper=14.0)
df["rsl"] = df["sl"] - df["dah"]
df["block"] = pd.factorize(df["geno"] + "_" + df["tiempo"].astype(str))[0]
LOT_A = {"91400028", "91400035", "ANTSON013", "CUNSBR048", "NDSCAR038"}
df["day"] = pd.factorize(df["geno"].isin(LOT_A).map({True: "A", False: "B"}) + df["tiempo"].astype(str))[0]
REG = ["dah", "rsl", "gqi_e2"]
Y = {t: df[t].to_numpy(np.float32) for t in REG}
C = df["calidad_idx"].to_numpy(np.int64)
# rasgos de laboratorio auxiliares del CNN-MTL, como en el notebook (61 rasgos de dataset_final.csv)
LABCOLS = [c for c in df.columns if c.startswith(("fq_", "esp_", "met_"))]
LABV = df[LABCOLS].to_numpy(np.float32); LABM = (~np.isnan(LABV)).astype(np.float32)


def outer_splits(scheme):
    n = np.arange(len(df))
    if scheme == "P1":
        g = df["block"].to_numpy(); u_ = np.unique(g); rng = np.random.default_rng(SEED); rng.shuffle(u_)
        f = pd.Series(g).map({v: i % 5 for i, v in enumerate(u_)}).to_numpy()
        return [(n[f != k], n[f == k]) for k in range(5)]
    g = {"P1_day": df["day"], "P2": df["tiempo"], "P3": df["geno"]}[scheme].to_numpy()
    return list(LeaveOneGroupOut().split(n, groups=g))


INNER_UNIT = {"P1": "block", "P1_day": "day", "P2": "block", "P3": "geno"}


def inner_split(train, seed, scheme):
    """Entrenamiento / validación / calibración dentro del entrenamiento, separando la misma unidad que el
    esquema deja fuera (P3: genotipos retenidos; P1_day: días; P1 y P2: bloques genotipo × tiempo), para que
    la calibración conformal imite la situación de prueba."""
    b = df[INNER_UNIT[scheme]].to_numpy()[train]; u_ = np.unique(b); rng = np.random.default_rng(seed); rng.shuffle(u_)
    k = max(1, round(0.15 * len(u_)))
    va, ca = set(u_[:k]), set(u_[k:2 * k])
    return train[~np.isin(b, list(va | ca))], train[np.isin(b, list(va))], train[np.isin(b, list(ca))]


def conformal_q(res):
    n = len(res); lev = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(np.abs(res), lev, method="higher"))


def baselines(tr, te, y):
    d, g = df["dah"].to_numpy(float), df["geno"].to_numpy()
    out = {"Trivial": np.full(len(te), y[tr].mean())}
    out["DAH only"] = LinearRegression().fit(d[tr, None], y[tr]).predict(d[te, None])
    mg = pd.Series(y[tr]).groupby(g[tr]).mean()
    out["Genotype only"] = pd.Series(g[te]).map(mg).fillna(y[tr].mean()).to_numpy()
    G = pd.get_dummies(pd.Categorical(g, categories=sorted(set(g))), dtype=float).to_numpy()
    Z = np.column_stack([d, G])
    p = LinearRegression().fit(Z[tr], y[tr]).predict(Z[te])
    unseen = ~np.isin(g[te], np.unique(g[tr]))
    p[unseen] = out["DAH only"][unseen]
    out["Genotype + DAH"] = p
    return out


# ---------------------------------------------------------------- CNN-MTL
class CnnMtl(nn.Module):
    def __init__(self, n_in, n_lab):
        super().__init__()
        h, p = CNN["hidden"], CNN["dropout"]
        self.trunk = nn.Sequential(nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p[0]),
                                   nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(p[1]))
        self.reg = nn.Linear(h // 2, len(REG))         # dah, rsl, gqi_e2 (estandarizadas)
        self.quality = nn.Linear(h // 2, 4)
        self.lab = nn.Linear(h // 2, n_lab)

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * CNN["input_noise"]
        z = self.trunk(x)
        return self.reg(z), self.quality(z), self.lab(z)


def fit_cnn(tr, va, seed):
    sc = StandardScaler().fit(XB[tr])
    ym = {t: (Y[t][tr].mean(), Y[t][tr].std()) for t in REG}
    lm = np.nanmean(np.where(LABM[tr] > 0, LABV[tr], np.nan), axis=0); ls = np.nanstd(np.where(LABM[tr] > 0, LABV[tr], np.nan), axis=0)
    lm, ls = np.nan_to_num(lm), np.nan_to_num(ls, nan=1.0); ls[ls == 0] = 1
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    def pack(i):
        yr = np.column_stack([(Y[t][i] - ym[t][0]) / ym[t][1] for t in REG])
        lv = np.where(LABM[i] > 0, (LABV[i] - lm) / ls, 0)
        return T(sc.transform(XB[i])), T(yr), T(C[i], torch.long), T(lv), T(LABM[i])
    Xt, Yt, Ct, Lt, Mt = pack(tr); Xv, Yv, Cv, Lv, Mv = pack(va)
    torch.manual_seed(seed); np.random.seed(seed)
    net = CnnMtl(XB.shape[1], LABV.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=CNN["learning_rate"], weight_decay=CNN["weight_decay"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=CNN["scheduler_factor"], patience=CNN["scheduler_patience"])
    def loss(o, y, c, l, mk):
        r, q, lb = o
        lab_l = ((lb - l) ** 2 * mk).sum() / mk.sum().clamp(min=1)
        return nn.functional.mse_loss(r, y) + nn.functional.cross_entropy(q, c, label_smoothing=CNN["label_smoothing"]) \
            + CNN["auxiliary_weight"] * lab_l
    best, state, stale = np.inf, None, 0
    for ep in range(2 if QUICK else CNN["max_epochs"]):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), CNN["batch_size"]):
            j = perm[i:i + CNN["batch_size"]]
            if len(j) < 2: continue
            opt.zero_grad(); loss(net(Xt[j]), Yt[j], Ct[j], Lt[j], Mt[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lv = float(loss(net(Xv), Yv, Cv, Lv, Mv))
        sch.step(lv)
        if lv < best - 1e-4:
            best, stale, state = lv, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= CNN["patience"]: break
    net.load_state_dict(state); net.eval()
    def predict(i):
        with torch.no_grad():
            r, q, _ = net(T(sc.transform(XB[i])))
        r = r.numpy()
        return {t: r[:, k] * ym[t][1] + ym[t][0] for k, t in enumerate(REG)}, torch.softmax(q, 1).numpy()
    return predict, ep + 1


# ---------------------------------------------------------------- validación
res = {"df": df[["ruta", "geno", "tiempo", "dah", "calidad_idx", "sl", "rsl", "gqi_e2", "block", "day"]],
       "alpha": ALPHA, "schemes": {}}
t0 = time.time()
for sc_name in SCHEMES:
    P = {"fold": np.full(len(df), -1)}
    Q = []
    for k, (train, test) in enumerate(outer_splits(sc_name)[:1] if QUICK else outer_splits(sc_name)):
        P["fold"][test] = k
        tr, va, ca = inner_split(train, SEED + k, sc_name)
        trva = np.concatenate([tr, va])
        q = {"fold": k, "n_cal": len(ca)}
        # CTD-GB: entrena con tr+va, calibra en ca
        for t in REG:
            mdl = HistGradientBoostingRegressor(**CTD_GB, random_state=SEED).fit(X[trva], Y[t][trva])
            P.setdefault(f"CTD-GB|{t}", np.full(len(df), np.nan))[test] = mdl.predict(X[test])
            q[f"CTD-GB|{t}"] = conformal_q(Y[t][ca] - mdl.predict(X[ca]))
        cla = HistGradientBoostingClassifier(**CTD_GB, random_state=SEED).fit(X[trva], C[trva])
        P.setdefault("CTD-GB|class", np.full(len(df), -1))[test] = cla.predict(X[test])
        # CNN-MTL: entrena con tr, parada temprana con va, calibra en ca
        predict, epochs = fit_cnn(tr, va, SEED)
        pr_te, pq_te = predict(test); pr_ca, _ = predict(ca)
        for t in REG:
            P.setdefault(f"CNN-MTL|{t}", np.full(len(df), np.nan))[test] = pr_te[t]
            q[f"CNN-MTL|{t}"] = conformal_q(Y[t][ca] - pr_ca[t])
        P.setdefault("CNN-MTL|class", np.full(len(df), -1))[test] = pq_te.argmax(1)
        q["cnn_epochs"] = epochs
        # referencias (metadatos verdaderos, no la foto), ajustadas con tr+va y calibradas en ca
        for t in REG:
            for name, v in baselines(trva, test, Y[t]).items():
                P.setdefault(f"{name}|{t}", np.full(len(df), np.nan))[test] = v
            for name, v in baselines(trva, ca, Y[t]).items():
                q[f"{name}|{t}"] = conformal_q(Y[t][ca] - v)
        mode = np.bincount(C[trva]).argmax()
        P.setdefault("Trivial|class", np.full(len(df), -1))[test] = mode
        Q.append(q)
        print(f"{sc_name} pliegue {k} ({len(test)} fotos de prueba, {len(ca)} de calibración, CNN {epochs} épocas) "
              f"{time.time() - t0:.0f} s", flush=True)
    res["schemes"][sc_name] = {"pred": P, "q": pd.DataFrame(Q)}
    pickle.dump(res, open(OUT, "wb"))
print("fin", time.time() - t0)
