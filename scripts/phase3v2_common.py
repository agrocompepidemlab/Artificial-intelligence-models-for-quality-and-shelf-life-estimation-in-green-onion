"""Fase 3 v2: funciones comunes de procesamiento de imagen (segmentación, corrección de color, recorte,
alineación y descriptores sin escala). Ninguna función usa etiquetas."""
import os, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, measure, morphology, transform
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

ROOT = Path(__file__).resolve().parents[1]                                   # el repositorio
DATA = Path(os.environ.get("GREEN_ONION_DATA", ROOT / "data" / "phase3"))               # dataset_final.csv, cache/tramos.csv
PHOTOS = Path(os.environ.get("GREEN_ONION_PHOTOS", DATA / "photos"))         # carpeta que contiene T1–T4
SEG_SIDE = 384          # resolución a la que se validó `segment`
WORK_DRAFT = (1200, 800)    # decodificación JPEG reducida (1/4 → 1296 × 864)
DESC_LEN = 640          # longitud a la que se normaliza la planta para los descriptores
CROP_W, CROP_H = 448, 112   # recorte para la red (múltiplos de 14 y 4:1)
NEUTRAL = np.array([0.485, 0.456, 0.406])   # gris "neutro" para la red (media de ImageNet)


def photo_path(ruta):
    parts = Path(str(ruta).replace(chr(92), "/")).parts
    return PHOTOS.joinpath(*parts[parts.index("GreenOnionCap3") + 1:])


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def segment(rgb):
    """Copia literal de la segmentación del notebook de la Fase 3 (sección 3)."""
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
        return np.zeros((h, w), bool), np.zeros((h, w), bool), table_colour
    table = ndi.binary_fill_holes(morphology.closing(labels == in_centre.argmax(), morphology.disk(25, decomposition='sequence')))
    onion = morphology.erosion(table, morphology.disk(6)) & (distance > 22)
    onion = morphology.opening(onion, morphology.disk(1))
    return morphology.remove_small_objects(onion, min_size=80), table, table_colour


def _quad_design(ys, xs, h, w):
    y, x = ys / h - 0.5, xs / w - 0.5
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


def segment_v2(rgb):
    """Segmentación v2 (a 384 px). 1) Mesa = región oscura y neutra mayor que toca el centro; su color se modela con
    una superficie cuadrática en L*, a*, b* (sigue el degradado de luz). 2) Extensión de la mesa = envolvente convexa
    de esa región (la mesa es un rectángulo en perspectiva), un poco reducida para no tocar su borde. 3) Planta =
    píxeles dentro de la mesa que se apartan del fondo modelado (ΔE > 20) y no son sombra (más oscuros y sin color)."""
    lab = color.rgb2lab(rgb)
    h, w = lab.shape[:2]
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    cand = morphology.opening((C < 10) & (L < 55), morphology.disk(2))
    lab_ = measure.label(cand)
    centre = np.zeros((h, w), bool); centre[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    n = np.bincount(lab_[centre].ravel(), minlength=lab_.max() + 1); n[0] = 0
    if lab_.max() == 0 or n.max() == 0:
        return np.zeros((h, w), bool), np.zeros((h, w), bool), np.zeros_like(lab), C, np.zeros((h, w))
    # la planta puede partir la mesa en dos: se unen todas las regiones grandes (≥ 15 % de la mayor) que tocan el centro
    size = np.bincount(lab_.ravel(), minlength=lab_.max() + 1); size[0] = 0
    big = np.flatnonzero((n > 0) & (size >= 0.15 * size[n.argmax()]))
    table0 = np.isin(lab_, big)
    ys, xs = np.nonzero(table0)
    sub = np.random.default_rng(0).choice(len(ys), min(len(ys), 6000), replace=False)
    A = _quad_design(ys[sub], xs[sub], h, w); Y = lab[ys[sub], xs[sub]]
    keep = np.ones(len(sub), bool)
    for _ in range(2):                                   # ajuste robusto: 2 pasadas quitando residuos grandes
        coef, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
        res = np.linalg.norm(Y - A @ coef, axis=1)
        keep = res < max(2.5 * np.median(res[keep]) * 1.4826, 3)
    YY, XX = np.mgrid[0:h, 0:w]
    bg = (_quad_design(YY.ravel(), XX.ravel(), h, w) @ coef).reshape(h, w, 3)
    d = np.linalg.norm(lab - bg, axis=2)
    table = measure.label(morphology.closing((d < 12) & (C < 12), morphology.disk(3)))
    tt = np.bincount(table[table0].ravel(), minlength=table.max() + 1); tt[0] = 0
    table = np.isin(table, np.flatnonzero(tt >= 0.15 * tt.max()))
    hull = morphology.convex_hull_image(table)
    hull = morphology.erosion(hull, morphology.disk(4))
    shadow = (L < bg[..., 0] + 2) & (C < 10)
    glare = (C < 8) & (L < bg[..., 0] + 28)          # reflejo: gris y poco más claro que la mesa
    onion = hull & (d > 20) & ~shadow & ~glare
    onion = morphology.opening(onion, morphology.disk(1))
    onion = morphology.remove_small_objects(onion, min_size=40)
    return onion, hull, bg, C, np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def clean_onion(onion, table, chroma, hue=None):
    """Quita de la máscara lo que no es planta.
    · Candidatos: componentes que no están pegados al borde de la mesa (≤ 50 % en la franja interior de 6 px) ni al
      marco de la foto (≤ 80 % en el 10 % exterior).
    · Componente principal: el que maximiza tamaño × cercanía al centro de la foto (peso gaussiano, σ = 0,25 de la
      semidiagonal) entre los que tienen color de planta (≥ 30 % de píxeles con C* ≥ 12 y tono 65–150°).
    · Fragmentos: se suman en cadena los que tienen color de planta, miden menos del doble del principal, no son
      objetos compactos sin color de planta (solidez > 0,85 y < 50 % de color de planta: tarjetas, papeles,
      lápices), no tienen su centro en el 5 % exterior de la foto, no están sobre el borde de la mesa (≤ 30 % en la
      franja de 8 px: ladrillos de la pared) y están a ≤ 10 % del ancho de la imagen de lo ya aceptado (≤ 4 % si son
      pequeños: < 10 % del principal).
    · En el 5 % exterior de la foto se quitan los píxeles pardos, anaranjados o rojos (pared, puerta, ladrillo).
    Las cebollas de otra bandeja al fondo, las tarjetas rotuladas y los objetos quedan en la periferia o son
    compactos; la planta fotografiada está hacia el centro y es filamentosa."""
    lab_ = measure.label(onion)
    if lab_.max() == 0:
        return onion
    h, w = onion.shape
    band = morphology.erosion(table, morphology.disk(5)) & ~morphology.erosion(table, morphology.disk(11))
    frame = np.ones_like(onion); frame[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)] = False
    K = lab_.max() + 1
    n = np.bincount(lab_.ravel(), minlength=K).astype(float); n[0] = 0
    share = lambda m: np.bincount(lab_[m].ravel(), minlength=K) / np.maximum(n, 1)
    s_band, s_frame, s_col = share(band), share(frame), share(chroma >= 12)
    s_plant = share((chroma >= 12) & (hue >= 65) & (hue <= 150)) if hue is not None else s_col
    yy, xx = np.mgrid[0:h, 0:w]
    cy = np.bincount(lab_.ravel(), weights=yy.ravel(), minlength=K) / np.maximum(n, 1) / h
    cx = np.bincount(lab_.ravel(), weights=xx.ravel(), minlength=K) / np.maximum(n, 1) / w
    d = np.hypot((cy - 0.5) * 2, (cx - 0.5) * 2) / np.sqrt(2)
    wcen = np.exp(-d ** 2 / (2 * 0.25 ** 2))
    solidity = np.zeros(K)
    for rp in measure.regionprops(lab_):
        if rp.area >= 50:
            solidity[rp.label] = rp.solidity
    ok = (s_band <= 0.5) & (s_frame <= 0.8); ok[0] = False
    score = np.where(ok & (s_plant >= 0.3), n * wcen, -1)
    if score.max() <= 0:
        score = np.where(ok, n * (s_col + 0.05) * wcen, -1)
    main = int(np.argmax(score)) if score.max() > 0 else int(np.argmax(n))
    kept = {main}
    compact_object = (solidity > 0.85) & (s_plant < 0.5)
    inner = (cy > 0.05) & (cy < 0.95) & (cx > 0.05) & (cx < 0.95)
    rim = table & ~morphology.erosion(table, morphology.disk(8))          # franja del borde de la mesa (≈ 2 % del ancho)
    s_rim = share(rim)
    cand = ok & (s_plant >= 0.3) & (n < 2 * n[main]) & ~compact_object & inner & (s_rim <= 0.3)
    small = n < 0.1 * n[main]
    while True:          # en cadena: fragmentos grandes a ≤ 10 % del ancho de lo ya aceptado; pequeños a ≤ 4 %
        dist = ndi.distance_transform_edt(~np.isin(lab_, list(kept)))
        reach = np.bincount(lab_.ravel(), weights=(dist <= 0.10 * w).ravel(), minlength=K) > 0
        close = np.bincount(lab_.ravel(), weights=(dist <= 0.04 * w).ravel(), minlength=K) > 0
        hit = set(np.flatnonzero(cand & np.where(small, close, reach)))
        if hit <= kept:
            break
        kept |= hit
    out = np.isin(lab_, list(kept))
    if hue is not None:            # pared, puerta o ladrillo pegados a una punta en el 5 % exterior de la foto
        edge = np.ones_like(onion); edge[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)] = False
        out &= ~(edge & (chroma >= 12) & ((hue < 60) | (hue > 160)))
    return out


def load(ruta, draft=WORK_DRAFT, data=None):
    """Abre la foto (o sus bytes ya leídos, para leer el disco externo en orden con un solo hilo)."""
    import io
    img = Image.open(io.BytesIO(data) if data is not None else photo_path(ruta)); img.draft("RGB", draft)
    return img.convert("RGB")


def small_rgb(img, side=SEG_SIDE):
    s = img.copy(); s.thumbnail((side, side))
    return np.asarray(s, dtype=np.float64) / 255.0


LUT_LIN = np.where(np.arange(256) / 255 <= 0.04045, np.arange(256) / 255 / 12.92,
                   ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4).astype(np.float32)
_M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]], np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], np.float32)


def lab_from_linear(lin):
    """CIELAB (D65) desde RGB lineal; mismas constantes que skimage.color.rgb2lab, en float32."""
    t = (lin.astype(np.float32) @ _M.T) / _WHITE
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def stage_a(ruta, data=None):
    """Etapa A (sin etiquetas ni corrección): segmentación a 384 px, color mediano de la mesa (RGB lineal, sin la
    planta), máscara fina en la región de la planta, giro al eje mayor, recorte y escala a DESC_LEN de largo, base
    (extremo menos verde) a la izquierda. Devuelve el control de calidad y el recorte crudo con su máscara."""
    img = load(ruta, data=data)
    W, H = img.size
    s = img.copy(); s.thumbnail((SEG_SIDE, SEG_SIDE)); su8 = np.asarray(s)
    onion_s, hull_s, bg_s, C_s, H_s = segment_v2(su8 / 255.0)
    onion_s = clean_onion(onion_s, hull_s, C_s, H_s)
    free = hull_s & ~morphology.dilation(onion_s, morphology.disk(8, decomposition='sequence'))
    med = np.median(LUT_LIN[su8][free], axis=0) if free.sum() > 500 else np.full(3, np.nan)
    lab_ = measure.label(onion_s)
    areas = np.bincount(lab_.ravel())[1:] if lab_.max() else np.array([0])
    qc = {"tab_R": med[0], "tab_G": med[1], "tab_B": med[2], "onion_frac_384": onion_s.mean(), "table_frac_384": hull_s.mean(),
          "n_comp_384": int(lab_.max()), "largest_comp_share": float(areas.max() / max(areas.sum(), 1))}
    if onion_s.sum() < 30:
        return qc, None
    zone_s = morphology.dilation(onion_s, morphology.disk(3))
    ys, xs = np.nonzero(zone_s)
    sy0, sy1, sx0, sx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fy, fx = H / su8.shape[0], W / su8.shape[1]
    x0, x1, y0, y1 = int(sx0 * fx), min(W, int(np.ceil(sx1 * fx))), int(sy0 * fy), min(H, int(np.ceil(sy1 * fy)))
    f = min(1.0, 900 / max(x1 - x0, y1 - y0))
    rw, rh = max(8, round((x1 - x0) * f)), max(8, round((y1 - y0) * f))
    reg = np.asarray(img.crop((x0, y0, x1, y1)).resize((rw, rh), Image.BILINEAR))
    zone = np.asarray(Image.fromarray(zone_s[sy0:sy1, sx0:sx1]).resize((rw, rh), Image.NEAREST))
    bg = np.stack([np.asarray(Image.fromarray(bg_s[sy0:sy1, sx0:sx1, k].astype(np.float32)).resize((rw, rh), Image.BILINEAR))
                   for k in range(3)], axis=-1)
    lab = lab_from_linear(LUT_LIN[reg])
    Cc = np.hypot(lab[..., 1], lab[..., 2])
    shadow = (lab[..., 0] < bg[..., 0] + 2) & (Cc < 10)
    glare = (Cc < 8) & (lab[..., 0] < bg[..., 0] + 28)
    fine = (np.linalg.norm(lab - bg, axis=2) > 20) & ~shadow & ~glare & zone
    fine = morphology.remove_small_objects(morphology.opening(fine, morphology.disk(1)), min_size=int(40 * (rw / (sx1 - sx0)) ** 2))
    if fine.sum() < 200:
        return qc, None
    yy, xx = np.nonzero(fine)
    ev, vec = np.linalg.eigh(np.cov(np.vstack([xx, yy])))
    angle = float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1])))
    r = np.asarray(Image.fromarray(reg).rotate(angle, resample=Image.BILINEAR, expand=True))
    m = np.asarray(Image.fromarray(fine.astype(np.uint8) * 255).rotate(angle, resample=Image.NEAREST, expand=True)) > 127
    yy, xx = np.nonzero(m)
    r, m = r[yy.min():yy.max() + 1, xx.min():xx.max() + 1], m[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    len_px = m.shape[1] / f            # largo de la planta en píxeles de la foto a 1/4
    Hn = max(8, round(m.shape[0] * DESC_LEN / m.shape[1]))
    r = np.asarray(Image.fromarray(np.ascontiguousarray(r)).resize((DESC_LEN, Hn), Image.BILINEAR))
    m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((DESC_LEN, Hn), Image.NEAREST)) > 127
    a_ = lab_from_linear(LUT_LIN[r])[..., 1]
    q = max(1, int(0.15 * DESC_LEN))
    nong = (a_ > -6) & m
    left = nong[:, :q].sum() / max(m[:, :q].sum(), 1); right = nong[:, -q:].sum() / max(m[:, -q:].sum(), 1)
    flipped = bool(right > left)
    if flipped:
        r, m = r[:, ::-1], m[:, ::-1]
    qc.update({"mask_px_region": int(fine.sum()), "angle": angle, "flipped": flipped, "plant_len_px": len_px,
               "plant_h_over_len": Hn / DESC_LEN, "region_scale": f})
    return qc, (np.ascontiguousarray(r), np.ascontiguousarray(m))


def stage_b(r, m, gain):
    """Etapa B: corrección de color de la sesión (ganancia por canal en RGB lineal), descriptores sin escala y
    recorte para la red. También el color crudo y corregido de la base y la punta (validación con colorímetro)."""
    lin_raw = LUT_LIN[r]
    lin = np.clip(lin_raw * gain.astype(np.float32), 0, 1)
    lab_c, lab_r = lab_from_linear(lin), lab_from_linear(lin_raw)
    f = descriptors(lab_c, m)
    q = max(1, int(0.3 * m.shape[1]))
    for tag, LB in [("raw", lab_r), ("cor", lab_c)]:
        for part, sl in [("base", slice(0, q)), ("tip", slice(-q, None))]:
            px = LB[:, sl][m[:, sl]]
            f[f"{tag}_{part}_L"] = px[:, 0].mean() if len(px) else np.nan
            f[f"{tag}_{part}_hue"] = (np.degrees(np.arctan2(px[:, 2].mean(), px[:, 1].mean())) % 360) if len(px) else np.nan
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return f, crop_for_net(srgb, m)


def crop_for_net(r, m):
    """Planta sobre gris neutro, con relleno hasta 4:1 y reducida a 448 × 112."""
    x = r.copy(); x[~m] = NEUTRAL
    h, w = m.shape
    if w / h >= 4:
        H, W = int(np.ceil(w / 4)), w
    else:
        H, W = h, 4 * h
    canvas = np.tile(NEUTRAL, (H, W, 1)).astype(np.float64)
    oy, ox = (H - h) // 2, (W - w) // 2
    canvas[oy:oy + h, ox:ox + w] = x
    out = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).resize((CROP_W, CROP_H), Image.BILINEAR)
    return np.asarray(out, dtype=np.uint8)


HUE_CLASSES = {  # fracciones de color fijadas a priori (Lab corregido; h en grados, C = croma)
    "green": lambda L, C, h: (C >= 15) & (h >= 100) & (h < 160),
    "yellow_green": lambda L, C, h: (C >= 15) & (h >= 85) & (h < 100),
    "yellow": lambda L, C, h: (C >= 15) & (h >= 60) & (h < 85),
    "brown": lambda L, C, h: (C >= 8) & (h < 60) | ((L < 45) & (C >= 8) & (h < 85)),
    "pale": lambda L, C, h: (C < 15) & (L >= 55),
    "dark": lambda L, C, h: (L < 30),
}


def descriptors(lab, m):
    """≈100 descriptores sin escala de la planta alineada (base a la izquierda), con Lab corregido a DESC_LEN de largo."""
    f = {}
    H = m.shape[0]
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b); hue = np.degrees(np.arctan2(b, a)) % 360
    for name, v in [("L", L), ("a", a), ("b", b), ("C", C), ("h", hue)]:
        x = v[m]
        f[f"{name}_mean"], f[f"{name}_sd"] = x.mean(), x.std()
        f[f"{name}_p10"], f[f"{name}_p50"], f[f"{name}_p90"] = np.percentile(x, [10, 50, 90])
    for name, rule in HUE_CLASSES.items():
        f[f"frac_{name}"] = rule(L, C, hue)[m].mean()
    area = m.sum()
    edges = np.linspace(0, DESC_LEN, 11).astype(int)
    for k in range(10):
        mm = m[:, edges[k]:edges[k + 1]]
        cols = mm.any(0)
        f[f"prof{k}_width"] = (mm.sum(0)[cols].mean() / DESC_LEN) if cols.any() else 0.0
        for name, v in [("L", L), ("a", a), ("b", b)]:
            vv = v[:, edges[k]:edges[k + 1]][mm]
            f[f"prof{k}_{name}"] = vv.mean() if vv.size else np.nan
    rp = measure.regionprops(m.astype(np.uint8))[0]
    f["area_over_len2"] = area / DESC_LEN ** 2
    f["height_over_len"] = H / DESC_LEN
    f["solidity"], f["extent"] = rp.solidity, rp.extent
    f["n_components"] = measure.label(m).max()
    sk = morphology.skeletonize(m)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    f["n_endpoints"] = int(((nb == 1) & sk).sum())
    f["skeleton_over_len"] = sk.sum() / DESC_LEN
    # textura en L* dentro de la máscara
    q = np.clip((L / 100 * 31).astype(int) + 1, 1, 32); q[~m] = 0
    g = graycomatrix(q.astype(np.uint8), distances=[1, 3], angles=[0, np.pi / 2], levels=33, symmetric=True)
    g = g[1:, 1:].astype(float); g /= g.sum(axis=(0, 1), keepdims=True).clip(min=1)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        v = graycoprops(g, prop)
        f[f"glcm_{prop}_d1"], f[f"glcm_{prop}_d3"] = v[0].mean(), v[1].mean()
    lbp = local_binary_pattern((L / 100 * 255).astype(np.uint8), 8, 1, "uniform")
    hist = np.bincount(lbp[m].astype(int), minlength=10)[:10] / max(area, 1)
    for k in range(10):
        f[f"lbp{k}"] = hist[k]
    gy, gx = np.gradient(L)
    f["grad_mean"] = np.hypot(gx, gy)[m].mean()
    return f
