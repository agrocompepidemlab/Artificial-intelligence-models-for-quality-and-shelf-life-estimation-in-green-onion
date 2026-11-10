"""Fase 3 — reencuadre: esquemas de validación (fuga), referencias por DAH y genotipo, bootstrap por
conglomerados y evaluación de la decisión. Prototipo con CTD-GB antes de pasarlo al notebook."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]                       # el repositorio
DATA = Path(os.environ.get("GREEN_ONION_DATA", ROOT / "data"))   # dataset_final.csv y cachés de rasgos
PHOTOS = Path(os.environ.get("GREEN_ONION_PHOTOS", DATA / "photos"))   # carpeta que contiene T1–T4
LAB_FILE = ROOT / "data" / "Phase_2_Physicochemical.xlsx"


def photo_path(ruta):
    """La foto en este equipo: la ruta guardada en dataset_final.csv se reubica en PHOTOS."""
    parts = Path(str(ruta).replace(chr(92), "/")).parts
    return PHOTOS.joinpath(*parts[parts.index("GreenOnionCap3") + 1:])

import sys, time, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression

OUT = sys.argv[1]
SEED = 42
CTD_GB = dict(max_iter=400, learning_rate=0.05, max_depth=7, l2_regularization=1.0)

# vida útil por supervivencia (Fase 1, encuesta en línea, t50 con censura por intervalo; tope 14 d = último tiempo observado)
SL_SURV = {"CUNSBR048": 12.80, "91400070": 10.14, "ANTSNC003": 15.35, "91400028": 12.31, "NDSCAR038": 5.96,
           "91400035": 5.37, "ANTSON013": 6.29, "ANTSRO029": 14.14, "91400071": 8.38, "ANTSRO030": 11.29}
SL_CAP = 14.0

df = pd.read_csv(DATA / "dataset_final.csv")
assert df["vida_util_restante_dias"].notna().all()
X = np.load(DATA / "cache_descriptores_mejorados.npz", allow_pickle=True)["X"].astype(np.float32)
X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
emb = np.load(DATA / "cache_embeddings_cnn.npz")["X"].astype(np.float32)

df["geno"] = df["genotipo"].astype(str).replace({"CUNSBRO048": "CUNSBR048"})
gq = pd.read_csv(ROOT / "results" / "GreenOQ_results.csv")
gq["Genotype"] = gq["Genotype"].astype(str)
df["gqi_new"] = df.set_index(["geno", "tiempo"]).index.map(dict(zip(zip(gq["Genotype"], gq["Time"]), gq["GQI"])))
assert df["gqi_new"].notna().all()
# GQI del experimento 2 (las fotos son de ese experimento): misma fórmula de la Fase 2 sobre las medias de la celda
lab = pd.read_excel(LAB_FILE)
V = ["Pseudosteam length (cm)", "Pseudosteam Diameter (cm)", "Pseudosteam Hue", "Leave Hue", "Firmness (N)"]
W = np.array([74.5, 72.5, 75, 73, 84]); W = W / W.sum()
m = lab[lab["Experimento"] == 2].groupby(["Genotipo", "Tiempo"])[V].mean()
u = np.column_stack([np.clip((m[V[0]] - 15) / 15, 0, 1), np.clip((m[V[1]] - 5) / 18, 0, 1),
                     np.exp(-(m[V[2]] - 102.9) ** 2 / (2 * 16 ** 2)),
                     np.exp(-np.minimum(0, m[V[3]] - 124.1) ** 2 / (2 * 5.2 ** 2)), np.clip(1 - (m[V[4]] - 3) / 27, 0, 1)])
g2 = pd.Series(100 * (u * W).sum(1), index=m.index)
df["gqi_e2"] = [g2[(g, t)] for g, t in zip(df["geno"], df["tiempo"])]
df["sl"] = df["geno"].map(SL_SURV).clip(upper=SL_CAP)
df["rsl"] = df["sl"] - df["dah"]
df["reject"] = (df["rsl"] <= 0).astype(int)

# ---- unidades de agrupación
folder = df["ruta"].str.rsplit("\\", n=1).str[0]
df["g_folder"] = pd.factorize(folder)[0]
df["g_block"] = pd.factorize(df["geno"] + "_" + df["tiempo"].astype(str))[0]
LOT_A = {"91400028", "91400035", "ANTSON013", "CUNSBR048", "NDSCAR038"}
df["g_day"] = pd.factorize(df["geno"].isin(LOT_A).map({True: "A", False: "B"}) + df["tiempo"].astype(str))[0]
# pseudotallo, como en el notebook: nueva unidad si la distancia coseno entre fotos consecutivas > 4 × mediana
shot = df["ruta"].str.extract(r"IMG_(\d+)")[0].astype(int).to_numpy()
unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
dist = np.full(len(df), np.nan); order = {}
for name, idx in df.groupby(folder).groups.items():
    idx = np.asarray(idx); idx = idx[np.argsort(shot[idx])]; order[name] = idx
    dist[idx[1:]] = 1 - np.sum(unit[idx[:-1]] * unit[idx[1:]], axis=1)
thr = np.nanmedian(dist) * 4
lab = np.empty(len(df), dtype=object)
for name, idx in order.items():
    b = 0
    for k, r in enumerate(idx):
        if k and dist[r] > thr: b += 1
        lab[r] = f"{name}__{b}"
df["g_pseudo"] = pd.factorize(lab)[0]
# pseudotallo = tramo contiguo dentro de la carpeta (segment_all.py); la persona confirmó que cada cambio de arreglo es otro pseudotallo
tramos = pd.read_csv(DATA / "cache" / "tramos.csv")
assert (tramos["ruta"].to_numpy() == df["ruta"].to_numpy()).all()
df["g_tramo"] = pd.factorize(tramos["tramo"])[0]
print("unidades:", {k: df[k].nunique() for k in ["g_pseudo", "g_tramo", "g_folder", "g_block", "g_day"]}, flush=True)

TARGETS = {"dah": "dah", "rsl": "rsl", "gqi_e2": "gqi_e2", "gqi_both": "gqi_new"}
y_cls = df["calidad_idx"].to_numpy()


def splits(scheme):
    n = np.arange(len(df))
    if scheme == "photo":
        return list(KFold(5, shuffle=True, random_state=SEED).split(n))
    if scheme in ("pseudo", "tramo", "folder", "block"):
        g = df[f"g_{scheme}"].to_numpy()
        rng = np.random.default_rng(SEED); u = np.unique(g); rng.shuffle(u)
        fold_of = {v: i % 5 for i, v in enumerate(u)}
        f = np.array([fold_of[v] for v in g])
        return [(n[f != k], n[f == k]) for k in range(5)]
    g = {"day": df["g_day"], "P2_time": df["tiempo"], "P3_genotype": df["geno"]}[scheme].to_numpy()
    return list(LeaveOneGroupOut().split(n, groups=g))


def baselines(tr, te, y):
    """Referencias que usan los metadatos verdaderos (no la foto)."""
    d, g = df["dah"].to_numpy(float), df["geno"].to_numpy()
    out = {}
    out["DAH only"] = LinearRegression().fit(d[tr, None], y[tr]).predict(d[te, None])
    m = pd.Series(y[tr]).groupby(g[tr]).mean()
    out["Genotype only"] = pd.Series(g[te]).map(m).fillna(y[tr].mean()).to_numpy()
    G = pd.get_dummies(pd.Categorical(g, categories=sorted(set(g))), dtype=float).to_numpy()
    seen = np.isin(g[te], np.unique(g[tr]))
    Z = np.column_stack([d, G])
    p = LinearRegression().fit(Z[tr], y[tr]).predict(Z[te])
    p[~seen] = out["DAH only"][~seen]          # genotipo no visto: solo queda el DAH
    out["Genotype + DAH"] = p
    return out


SCHEMES = sys.argv[2].split(",") if len(sys.argv) > 2 else \
    ["photo", "pseudo", "tramo", "folder", "block", "day", "P2_time", "P3_genotype"]
pred = {}
t0 = time.time()
for sc in SCHEMES:
    P = {f"{m}|{t}": np.full(len(df), np.nan) for t in TARGETS for m in ["CTD-GB", "DAH only", "Genotype only", "Genotype + DAH"]}
    P["CTD-GB|class"] = np.full(len(df), -1)
    P["fold"] = np.full(len(df), -1)
    for k, (tr, te) in enumerate(splits(sc)):
        P["fold"][te] = k
        for t, col in TARGETS.items():
            y = df[col].to_numpy(float)
            P[f"CTD-GB|{t}"][te] = HistGradientBoostingRegressor(**CTD_GB, random_state=SEED).fit(X[tr], y[tr]).predict(X[te])
            for m, v in baselines(tr, te, y).items():
                P[f"{m}|{t}"][te] = v
        P["CTD-GB|class"][te] = HistGradientBoostingClassifier(**CTD_GB, random_state=SEED).fit(X[tr], y_cls[tr]).predict(X[te])
        print(f"{sc} fold {k} listo ({time.time() - t0:.0f} s)", flush=True)
    pred[sc] = P
    pickle.dump({"df": df[["ruta", "geno", "tiempo", "dah", "calidad_idx", "sl", "rsl", "reject", "gqi_new", "gqi_e2",
                           "g_pseudo", "g_tramo", "g_folder", "g_block", "g_day"]], "pred": pred, "thr": thr}, open(OUT, "wb"))
print("fin", time.time() - t0)
