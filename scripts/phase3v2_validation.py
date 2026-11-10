"""Fase 3 v2 · validación (pliegues en paralelo). Uso:
  python run_validation_v2.py OUT.pkl [esquemas separados por coma] [--ladder]
Esquemas por defecto: P3, P4_lot, P1_day, P2 y P1 (5 repeticiones, semillas 42–46). Con --ladder: photo, pseudo, session."""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import phase3v2_models as M

OUT = sys.argv[1]
LADDER = "--ladder" in sys.argv
args = [a for a in sys.argv[2:] if not a.startswith("--")]
SCHEMES = args[0].split(",") if args else (["photo", "pseudo", "session"] if LADDER else ["P3", "P4_lot", "P1_day", "P2", "P1"])
tasks = []
for sc in SCHEMES:
    for sd in ([42, 43, 44, 45, 46] if sc == "P1" and not LADDER else [42]):
        for k in range(len(M.outer_splits(sc, sd))):
            tasks.append((sc, sd, k))
print(len(tasks), "pliegues", flush=True)
FOLDS = Path(OUT).with_suffix("").parent / ("folds_ladder" if LADDER else "folds_validation")
FOLDS.mkdir(exist_ok=True)


def fold_file(sc, sd, k):
    return FOLDS / f"{sc}_{sd}_{k}.pkl"


def run_and_save(sc, sd, k):
    """Cada pliegue se guarda en disco al terminar: si el proceso se interrumpe, se retoma desde ahí."""
    f = fold_file(sc, sd, k)
    if f.exists():
        return
    r = M.run_fold(sc, sd, k, LADDER)
    tmp = f.with_suffix(".tmp"); pickle.dump(r, open(tmp, "wb")); tmp.replace(f)


todo = [t for t in tasks if not fold_file(*t).exists()]
print(f"ya guardados: {len(tasks) - len(todo)} · por correr: {len(todo)}", flush=True)
t0 = time.time()
Parallel(n_jobs=7, verbose=10)(delayed(run_and_save)(sc, sd, k) for sc, sd, k in todo)
out = [pickle.load(open(fold_file(*t), "rb")) for t in tasks]
res = {"df": M.df[["ruta", "geno", "tiempo", "dah", "calidad_idx", "sl", "rsl", "gqi_e2", "block", "day", "lot", "session", "pseudo"]],
       "ok": M.OKP, "alpha": M.ALPHA, "plan": "PLAN_ANALISIS_v2.md", "hyper": {"M1": M.M1, "ridge_alphas": M.RIDGE_ALPHAS, "logit_C": M.LOGIT_C},
       "schemes": {}}
for (sc, sd, k), (test, P, q) in zip(tasks, out):
    key = sc if sd == 42 else f"{sc}#{sd}"
    S = res["schemes"].setdefault(key, {"pred": {"fold": np.full(M.N, -1)}, "q": []})
    S["pred"]["fold"][test] = k
    for kk, v in P.items():
        shape = (M.N, 4) if np.ndim(v) == 2 else M.N
        S["pred"].setdefault(kk, np.full(shape, np.nan))[test] = v
    S["q"].append(q)
for S in res["schemes"].values():
    S["q"] = pd.DataFrame(S["q"]).sort_values("fold").reset_index(drop=True)
pickle.dump(res, open(OUT, "wb"))
print("fin", (time.time() - t0) / 60, "min", flush=True)
