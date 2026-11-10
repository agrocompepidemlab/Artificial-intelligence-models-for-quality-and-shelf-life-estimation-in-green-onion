"""Fase 3 v2 · une las sesiones procesadas en el orden de dataset_final.csv (los recortes se escriben directo en
crops.npy, sin cargarlos todos en memoria) y calcula el embedding ResNet18 (ImageNet, congelada) de cada recorte
alineado de la planta (448 × 112), promedio de 4 volteos (original, horizontal, vertical y ambos)."""
import sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torchvision import models
import phase3v2_common as V

CACHE = V.Path(sys.argv[1])
torch.set_num_threads(8)
df = pd.read_csv(V.DATA / "dataset_final.csv", usecols=["ruta"])
pos = pd.Series(np.arange(len(df)), index=df["ruta"])
files = sorted((CACHE / "sessions").glob("s*.npz"))
N = len(df)
crops = np.lib.format.open_memmap(CACHE / "crops.npy", mode="w+", dtype=np.uint8, shape=(N, V.CROP_H, V.CROP_W, 3))
metas, D, seen = [], None, np.zeros(N, bool)
for f in files:
    z = np.load(f); m = pd.read_csv(f.with_suffix(".csv"))
    idx = pos.reindex(m["ruta"]).to_numpy(); assert not np.isnan(idx).any(); idx = idx.astype(int)
    if D is None:
        names = z["names"]; D = np.full((N, len(names)), np.nan, np.float32)
    D[idx] = z["D"]; crops[idx] = z["crops"]; seen[idx] = True
    metas.append(m.assign(_pos=idx))
assert seen.all(), f"faltan {(~seen).sum()} fotos"
crops.flush()
meta = pd.concat(metas).sort_values("_pos").drop(columns="_pos").reset_index(drop=True)
assert (meta["ruta"].to_numpy() == df["ruta"].to_numpy()).all()
# control de calidad (sin etiquetas, fijado antes de validar): planta < 30 % del largo mediano = fallo de segmentación
med = meta.loc[meta["ok"], "plant_len_px"].median()
meta["qc_short"] = meta["ok"] & (meta["plant_len_px"] < 0.3 * med)
meta["ok"] = meta["ok"] & ~meta["qc_short"]
print("fotos", N, "· sin planta", int((~meta["ok"]).sum()), "· de ellas excluidas por largo < 30 % de la mediana:",
      int(meta["qc_short"].sum()), flush=True)

net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); net.fc = nn.Identity(); net.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
C = np.load(CACHE / "crops.npy", mmap_mode="r")
E = np.zeros((N, 512), np.float32)
t0 = time.time(); B = 64
with torch.no_grad():
    for s in range(0, N, B):
        x = (torch.from_numpy(np.ascontiguousarray(C[s:s + B])).permute(0, 3, 1, 2).float() / 255 - mean) / std
        E[s:s + B] = (sum(net(v) for v in [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]) / 4).numpy()
        if (s // B) % 50 == 0:
            print(f"{s + len(x)} / {N} ({time.time() - t0:.0f} s)", flush=True)
np.savez_compressed(CACHE / "features_v2.npz", D=D, names=names, E=E)
meta.to_csv(CACHE / "meta_v2.csv", index=False)
print("fin", time.time() - t0)
