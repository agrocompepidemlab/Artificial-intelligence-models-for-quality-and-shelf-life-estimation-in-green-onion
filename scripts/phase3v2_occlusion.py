"""Fase 3 v2 · mapas de oclusión del modelo de red (M2) en P3 (genotipo no visto).
Para 8 fotos de prueba (2 por DAH, de genotipos distintos) se reentrena el Ridge del DAH sin el genotipo de la foto
(igual que en el pliegue P3), y se tapa el recorte con un cuadro gris neutro de 28 × 28 px (paso 14 px): el cambio
de la predicción en cada posición mide cuánto aporta esa zona. También se resume, para 400 fotos de prueba al azar,
la importancia de la base (30 % izquierdo del recorte alineado), el centro y la punta."""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from torchvision import models
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import phase3v2_common as V

CACHE = V.Path(sys.argv[1]); OUT = sys.argv[2]
torch.set_num_threads(8)
df = pd.read_csv(V.DATA / "dataset_final.csv", usecols=["ruta", "genotipo", "dah", "calidad_idx"])
df["geno"] = df["genotipo"].astype(str).replace({"CUNSBRO048": "CUNSBR048"})
df["session"] = pd.factorize(df["ruta"].str.rsplit("\\", n=1).str[0])[0]
z = np.load(CACHE / "features_v2.npz", allow_pickle=True); E = z["E"]
meta = pd.read_csv(CACHE / "meta_v2.csv"); ok = meta["ok"].to_numpy(bool)
crops = np.load(CACHE / "crops.npy", mmap_mode="r")
SW = 1.0 / df.groupby("session")["ruta"].transform("size").to_numpy(float)
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0     # regularización elegida en la validación (mediana de P3)

net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); net.fc = nn.Identity(); net.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def embed(x):          # x: (n, H, W, 3) uint8 → embedding promedio de 4 volteos
    t = (torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 255 - mean) / std
    with torch.no_grad():
        return (sum(net(v) for v in [t, t.flip(3), t.flip(2), t.flip(2).flip(3)]) / 4).numpy()


rng = np.random.default_rng(7)
genos = sorted(df["geno"].unique())
picked = []
for dah in [2, 6, 10, 14]:
    for g in rng.choice(genos, 2, replace=False):
        cand = np.flatnonzero((df["dah"] == dah) & (df["geno"] == g) & ok)
        picked.append(int(rng.choice(cand)))
P, S = 28, 14
neutral = (np.array(V.NEUTRAL) * 255).astype(np.uint8)
maps, preds, models_cache = [], [], {}
for i in picked:
    g = df["geno"].iloc[i]
    if g not in models_cache:
        tr = np.flatnonzero((df["geno"] != g).to_numpy() & ok)
        sc = StandardScaler().fit(E[tr])
        models_cache[g] = (sc, Ridge(alpha=ALPHA).fit(sc.transform(E[tr]), df["dah"].to_numpy(float)[tr], sample_weight=SW[tr]))
    sc, rd = models_cache[g]
    base = crops[i].copy()
    p0 = rd.predict(sc.transform(embed(base[None])))[0]
    ys = range(0, V.CROP_H - P + 1, S); xs = range(0, V.CROP_W - P + 1, S)
    batch, pos = [], []
    for y in ys:
        for x in xs:
            b = base.copy(); b[y:y + P, x:x + P] = neutral; batch.append(b); pos.append((y, x))
    pr = rd.predict(sc.transform(embed(np.stack(batch))))
    heat = np.zeros((V.CROP_H, V.CROP_W)); cnt = np.zeros_like(heat)
    for (y, x), v in zip(pos, pr):
        heat[y:y + P, x:x + P] += abs(p0 - v); cnt[y:y + P, x:x + P] += 1
    maps.append(heat / np.maximum(cnt, 1)); preds.append(p0)
    print("foto", i, "DAH", df["dah"].iloc[i], "predicción", round(p0, 1), flush=True)

# resumen: importancia por tercios del recorte en 120 fotos de prueba (columnas tapadas de 1/10 del ancho)
share = []
sub = rng.choice(np.flatnonzero(ok), 120, replace=False)
for g in genos:
    idx = [i for i in sub if df["geno"].iloc[i] == g]
    if not idx:
        continue
    tr = np.flatnonzero((df["geno"] != g).to_numpy() & ok)
    sc = StandardScaler().fit(E[tr]); rd = Ridge(alpha=ALPHA).fit(sc.transform(E[tr]), df["dah"].to_numpy(float)[tr], sample_weight=SW[tr])
    for i in idx:
        base = crops[i].copy(); p0 = rd.predict(sc.transform(embed(base[None])))[0]
        cols = np.linspace(0, V.CROP_W, 11).astype(int); batch = []
        for k in range(10):
            b = base.copy(); b[:, cols[k]:cols[k + 1]] = neutral; batch.append(b)
        d = np.abs(p0 - rd.predict(sc.transform(embed(np.stack(batch)))))
        share.append(d / max(d.sum(), 1e-9))
share = np.array(share)
np.savez_compressed(OUT, maps=np.array(maps), preds=np.array(preds), picked=np.array(picked),
                    imgs=np.stack([crops[i] for i in picked]), column_share=share, alpha=ALPHA)
print("importancia media por décimo del recorte (base → punta):", np.round(share.mean(0), 3))
