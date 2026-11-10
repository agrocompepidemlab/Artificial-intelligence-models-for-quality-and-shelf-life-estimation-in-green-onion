"""Fase 3 v2 · procesamiento de todas las fotos, sesión por sesión (reanudable).
Por sesión: etapa A en cada foto (segmentación, color de la mesa, recorte alineado crudo); ganancia por canal en RGB
lineal que lleva la mediana de la mesa de la sesión a un gris de referencia fijo; etapa B (corrección, descriptores,
recorte para la red)."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import phase3v2_common as V

OUT = V.Path(sys.argv[1]); (OUT / "sessions").mkdir(parents=True, exist_ok=True)
NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 7
REF = np.full(3, 0.135)          # gris de referencia (RGB lineal), constante fijada a priori
df = pd.read_csv(V.DATA / "dataset_final.csv", usecols=["ruta"])
df["session"] = df["ruta"].str.rsplit("\\", n=1).str[0]
sessions = list(dict.fromkeys(df["session"]))


def none():
    return None


def read_session(si):
    """Un solo hilo lee los archivos de la sesión en orden (el disco externo es lento con lecturas en paralelo)."""
    rutas = df.loc[df["session"] == sessions[si], "ruta"].tolist()
    return rutas, [open(V.photo_path(r), "rb").read() for r in rutas]


todo = [si for si in range(len(sessions)) if not (OUT / "sessions" / f"s{si:03d}.npz").exists()]
t0 = time.time(); done = 0
reader = ThreadPoolExecutor(max_workers=1)
nxt = reader.submit(read_session, todo[0]) if todo else None
with Parallel(n_jobs=NJ) as par:
    for j, si in enumerate(todo):
        f = OUT / "sessions" / f"s{si:03d}.npz"
        rutas, blobs = nxt.result()
        nxt = reader.submit(read_session, todo[j + 1]) if j + 1 < len(todo) else None
        A = par(delayed(V.stage_a)(r, b) for r, b in zip(rutas, blobs))
        del blobs
        qc = pd.DataFrame([a[0] for a in A])
        med = np.nanmedian(qc[["tab_R", "tab_G", "tab_B"]].to_numpy(float), axis=0)
        gain = REF / med if np.isfinite(med).all() else np.ones(3)
        B = par(delayed(V.stage_b)(a[1][0], a[1][1], gain) if a[1] is not None else delayed(none)() for a in A)
        ok = [b is not None for b in B]
        names = list(next(b for b in B if b is not None)[0].keys())
        D = np.array([[b[0][k] for k in names] if b is not None else [np.nan] * len(names) for b in B], np.float32)
        crops = np.stack([b[1] if b is not None else np.zeros((V.CROP_H, V.CROP_W, 3), np.uint8) for b in B])
        meta = pd.concat([pd.DataFrame({"ruta": rutas, "ok": ok}), qc], axis=1)
        meta["gain_R"], meta["gain_G"], meta["gain_B"] = gain
        np.savez(f, D=D, names=np.array(names), crops=crops)
        meta.to_csv(OUT / "sessions" / f"s{si:03d}.csv", index=False)
        done += 1
        el = (time.time() - t0) / 60
        print(f"sesión {si + 1}/{len(sessions)} · {len(rutas)} fotos · sin planta {len(rutas) - sum(ok)} · ganancia {np.round(gain, 2)} "
              f"· {el:.1f} min · faltan ~{el / done * (len(todo) - j - 1):.0f} min", flush=True)
print("fin", (time.time() - t0) / 60, "min")
