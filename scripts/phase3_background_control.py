"""Control de atajo: rasgos del FONDO solamente (fuera de la máscara del pseudotallo).
1) estadísticos de color del fondo (Lab, global y en una rejilla 3x3), color mediano de la mesa;
2) embedding ResNet18 de la foto con la planta borrada (rellena con el color mediano de la mesa).
La segmentación es la misma función `segment` del notebook de la Fase 3."""
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
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage import color, measure, morphology
from torchvision import models, transforms
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT, OUT_MODELS = sys.argv[1], sys.argv[2]
SIDE = 384


def segment(rgb):            # copia literal del notebook (sección 3)
    lab = color.rgb2lab(rgb)
    h, w = lab.shape[:2]
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    cy, cx = h // 2, w // 2
    centre = (slice(cy - h // 3, cy + h // 3), slice(cx - w // 3, cx + w // 3))
    Lc, cc = L[centre], chroma[centre]
    candidates = (Lc < np.percentile(Lc, 55)) & (cc < 8)
    if candidates.sum() < 50:
        candidates = Lc < np.percentile(Lc, 40)
    table_colour = np.median(lab[centre][candidates], axis=0)
    distance = np.linalg.norm(lab - table_colour, axis=2)
    labels = measure.label(morphology.opening(distance < 14, morphology.disk(3)))
    in_centre = np.bincount(labels[centre].ravel(), minlength=labels.max() + 1)
    in_centre[0] = 0
    if in_centre.max() == 0:
        return np.zeros((h, w), bool), np.zeros((h, w), bool)
    table = ndi.binary_fill_holes(morphology.closing(labels == in_centre.argmax(), morphology.disk(25, decomposition='sequence')))
    onion = morphology.erosion(table, morphology.disk(6)) & (distance > 22)
    onion = morphology.opening(onion, morphology.disk(1))
    return morphology.remove_small_objects(onion, min_size=80), table


def one(path):
    p = photo_path(path)
    img = Image.open(p); img.draft("RGB", (2 * SIDE, 2 * SIDE))    # decodificación JPEG a escala reducida
    img = img.convert("RGB"); img.thumbnail((SIDE, SIDE))
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    onion, table = segment(rgb)
    lab = color.rgb2lab(rgb)
    # A. color del fondo: todo lo que no es planta (máscara del notebook ampliada 12 px)
    bg = ~morphology.binary_dilation(onion, morphology.disk(12, decomposition='sequence'))
    f = {}
    for k, ch in enumerate("Lab"):
        v = lab[..., k][bg]
        f[f"bg_{ch}_mean"], f[f"bg_{ch}_sd"] = v.mean(), v.std()
        f[f"bg_{ch}_p10"], f[f"bg_{ch}_p90"] = np.percentile(v, 10), np.percentile(v, 90)
        tv = lab[..., k][table & bg]
        f[f"table_{ch}_median"] = np.median(tv) if tv.size else np.nan
        nv = lab[..., k][~table]
        f[f"room_{ch}_mean"] = nv.mean() if nv.size else np.nan
    h, w = bg.shape
    for i in range(3):
        for j in range(3):
            cell = (slice(i * h // 3, (i + 1) * h // 3), slice(j * w // 3, (j + 1) * w // 3))
            m = bg[cell]
            for k, ch in enumerate("Lab"):
                f[f"grid{i}{j}_{ch}"] = lab[..., k][cell][m].mean() if m.any() else np.nan
    f["frac_table"] = table.mean()
    # B. entorno sin planta: la mesa entera (con la planta encima) se pinta de su color mediano
    tbl = morphology.binary_dilation(table | onion, morphology.disk(4))
    fill = np.median(rgb[table & bg], axis=0) if (table & bg).any() else np.median(rgb.reshape(-1, 3), axis=0)
    erased = rgb.copy(); erased[tbl] = fill
    small = Image.fromarray((erased * 255).astype(np.uint8)).resize((224, 224))
    return f, np.asarray(small, dtype=np.uint8), {"onion_frac": float(onion.mean()), "table_frac": float(tbl.mean())}


df = pd.read_csv(DATA / "dataset_final.csv")
net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); net.fc = nn.Identity(); net.eval()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
feats, embs, geos = [], np.zeros((len(df), 512), np.float32), []
t0 = time.time(); CH = 256
for s in range(0, len(df), CH):
    res = Parallel(n_jobs=7)(delayed(one)(p) for p in df["ruta"].iloc[s:s + CH])
    feats += [r[0] for r in res]; geos += [r[2] for r in res]
    x = torch.tensor(np.stack([r[1] for r in res])).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        embs[s:s + len(res)] = net(norm(x)).numpy()
    print(f"{s + len(res)} / {len(df)}  ({time.time() - t0:.0f} s)", flush=True)
F = pd.DataFrame(feats)
np.savez_compressed(OUT, X=F.to_numpy(np.float32), names=np.array(F.columns), emb=embs,
                    geo=pd.DataFrame(geos).to_numpy(np.float32), geo_names=np.array(pd.DataFrame(geos).columns))
print("fin", time.time() - t0)


# ================================================ modelos que solo ven el fondo

BG, OUT = OUT, OUT_MODELS
SEED = 42
df = pd.read_csv(DATA / "dataset_final.csv")
df["geno"] = df["genotipo"].astype(str).replace({"CUNSBRO048": "CUNSBR048"})
z = np.load(BG, allow_pickle=True)
A = z["X"].astype(np.float32); A = np.where(np.isnan(A), np.nanmedian(A, axis=0), A)
Bm = z["emb"].astype(np.float32)
block = pd.factorize(df["geno"] + "_" + df["tiempo"].astype(str))[0]
LOT_A = {"91400028", "91400035", "ANTSON013", "CUNSBR048", "NDSCAR038"}
day = pd.factorize(df["geno"].isin(LOT_A).map({True: "A", False: "B"}) + df["tiempo"].astype(str))[0]
TARGETS = {"time": df["tiempo"].to_numpy(), "class": df["calidad_idx"].to_numpy(), "genotype": pd.factorize(df["geno"])[0]}
n = np.arange(len(df))


def splits(scheme):
    if scheme == "photo":
        return list(KFold(5, shuffle=True, random_state=SEED).split(n))
    if scheme == "session":
        u = np.unique(block); rng = np.random.default_rng(SEED); rng.shuffle(u)
        f = pd.Series(block).map({v: i % 5 for i, v in enumerate(u)}).to_numpy()
        return [(n[f != k], n[f == k]) for k in range(5)]
    return list(LeaveOneGroupOut().split(n, groups=day))


models = {"background colour": (A, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)),
          "environment embedding": (Bm, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1)))}
res = {"names": z["names"], "pred": {}}
t0 = time.time()
for fname, (Xf, make) in models.items():
    for sc in ["photo", "session", "day"]:
        for t, y in TARGETS.items():
            p = np.full(len(df), -1)
            for tr, te in splits(sc):
                p[te] = make().fit(Xf[tr], y[tr]).predict(Xf[te])
            res["pred"][(fname, sc, t)] = p
            print(f"{fname} · {sc} · {t}: exactitud {np.mean(p == y):.3f} ({time.time() - t0:.0f} s)", flush=True)
        pickle.dump(res, open(OUT, "wb"))
print("fin")
