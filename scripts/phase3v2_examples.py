"""Fase 3 v2 · fotos de ejemplo con su máscara final (Figs. S6a–S6b), para que el notebook no necesite las fotos.
Repite los pasos de `stage_a` hasta la máscara fina y la devuelve sobre la foto completa (reducida)."""
import sys
import numpy as np, pandas as pd
from PIL import Image
from skimage import morphology
import phase3v2_common as V

CACHE = V.Path(sys.argv[1])
GENO = "ANTSRO030"           # fijado antes de mirar los ejemplos
df = pd.read_csv(V.DATA / "dataset_final.csv", usecols=["ruta", "genotipo", "tiempo", "calidad_idx"])
df["geno"] = df["genotipo"].astype(str).replace({"CUNSBRO048": "CUNSBR048"})
meta = pd.read_csv(CACHE / "meta_v2.csv"); ok = meta["ok"].to_numpy(bool)


def full_mask(ruta):
    img = V.load(ruta); W, H = img.size
    s = img.copy(); s.thumbnail((V.SEG_SIDE, V.SEG_SIDE)); su8 = np.asarray(s)
    onion_s, hull_s, bg_s, C_s, H_s = V.segment_v2(su8 / 255.0)
    onion_s = V.clean_onion(onion_s, hull_s, C_s, H_s)
    zone_s = morphology.dilation(onion_s, morphology.disk(3))
    D = 768; f = D / W; dw, dh = D, round(H * f)
    disp = np.asarray(img.resize((dw, dh), Image.BILINEAR))
    zone = np.asarray(Image.fromarray(zone_s).resize((dw, dh), Image.NEAREST))
    bg = np.stack([np.asarray(Image.fromarray(bg_s[..., k].astype(np.float32)).resize((dw, dh), Image.BILINEAR)) for k in range(3)], -1)
    lab = V.lab_from_linear(V.LUT_LIN[disp]); C = np.hypot(lab[..., 1], lab[..., 2])
    shadow = (lab[..., 0] < bg[..., 0] + 2) & (C < 10); glare = (C < 8) & (lab[..., 0] < bg[..., 0] + 28)
    fine = (np.linalg.norm(lab - bg, axis=2) > 20) & ~shadow & ~glare & zone
    fine = morphology.remove_small_objects(morphology.opening(fine, morphology.disk(1)), min_size=int(40 * (dw / su8.shape[1]) ** 2))
    return disp, fine


idx = []
for t, c in [(1, 0), (4, 3)]:
    cand = np.flatnonzero((df["geno"] == GENO).to_numpy() & (df["tiempo"] == t).to_numpy() & (df["calidad_idx"] == c).to_numpy() & ok)
    idx.append(int(cand[len(cand) // 2]))
res = [full_mask(df["ruta"].iloc[i]) for i in idx]
h = min(r[0].shape[0] for r in res)
np.savez_compressed(CACHE / "examples_v2.npz", orig=np.stack([r[0][:h] for r in res]), mask=np.stack([r[1][:h] for r in res]),
                    idx=np.array(idx), genotype=GENO)
print("ejemplos:", [df["ruta"].iloc[i] for i in idx])
