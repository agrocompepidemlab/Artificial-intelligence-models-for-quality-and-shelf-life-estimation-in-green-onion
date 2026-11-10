"""Phase 3 v2 · Word tables in English (three-line format, caption above, notes below), built from the tables
that the notebook saves in cache_v2/tables_v2.pkl. Usage: python make_tables_p3v2_docx.py TABLES.pkl OUT.docx"""
import sys, pickle
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm

T = pickle.load(open(sys.argv[1], "rb")); OUT = sys.argv[2]
FONT, SIZE = "Times New Roman", 9

doc = Document()
st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(SIZE)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); st.paragraph_format.space_after = Pt(0)
s0 = doc.sections[0]
s0.orientation = WD_ORIENT.LANDSCAPE; s0.page_width, s0.page_height = Cm(29.7), Cm(21.0)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(s0, m, Cm(1.8))
USABLE = 29.7 - 3.6


def fmt(v, d=2):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        return f"{v:.{d}f}".replace("-", "−")
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    s = str(v)
    if s in ("", "nan"):
        return ""
    s = s.replace(" -", " −").replace("[-", "[−")
    return "−" + s[1:] if s.startswith("-") and len(s) > 1 and (s[1].isdigit() or s[1] == ".") else s


def borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders")
    for edge, val in (("top", top), ("bottom", bottom), ("left", None), ("right", None)):
        el = OxmlElement(f"w:{edge}")
        if val:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(val)); el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        b.append(el)
    tcPr.append(b)


def caption(num, text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"Table {num}. "); r.bold = True; p.add_run(text)


def notes(*lines):
    for i, l in enumerate(lines):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(3 if i == 0 else 0)
        r = p.add_run(l); r.font.size = Pt(SIZE - 1)


def table(df, widths=None, group_first=False, decimals=2):
    df = df.copy()
    header = [str(c) for c in df.columns]
    rows = [[fmt(v, decimals) for v in r] for r in df.itertuples(index=False)]
    if group_first:
        last = None
        for r in rows:
            if r[0] == last:
                r[0] = ""
            else:
                last = r[0]
    n = len(header)
    widths = widths or [USABLE / n] * n
    t = doc.add_table(rows=1 + len(rows), cols=n); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j])
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 or (j == 1 and group_first) else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val); r.font.size = Pt(SIZE); r.bold = i == 0
            borders(c, top=12 if i == 0 else (6 if i == 1 else None), bottom=12 if i == len(rows) else None)
        if i == 0:
            h = OxmlElement("w:tblHeader"); t.rows[0]._tr.get_or_add_trPr().append(h)


title = doc.add_paragraph(); r = title.add_run("Phase 3 — Table 1 and supplementary tables S1–S14"); r.bold = True; r.font.size = Pt(12)
ABBR = ("DAH = days after harvest; P1 = genotype × time blocks (sessions) left out; P1 · capture day = capture days left out; "
        "P2 = storage time left out; P3 = genotype left out; P4 = one of the two photo groups left out (five genotypes photographed on "
        "their own days), i.e. unseen genotypes and photo days.")
CI = ("95% confidence interval from a cluster bootstrap (2000 resamples of genotype × time blocks in P1, P1 · capture day and P2; "
      "of genotypes in P3); P4 has two photo groups and no interval.")

caption("1", "Validation of the ensemble (plant only) against the reference that matters for each output, with sessions (P1), capture "
        "days, storage times (P2), genotypes (P3) and groups of genotypes photographed on other days (P4) left out. Continuous outputs "
        "are evaluated per genotype × time cell (mean prediction of its photographs, 40 cells) and the visual class per session (160 sessions).")
table(T["1"], widths=[4.3, 3.0, 1.9] + [(USABLE - 9.2) / 5] * 5, group_first=True)
notes("Values: R², accuracy or quadratic-weighted κ with the " + CI,
      "Ensemble = mean of CTD-GB v2 (99 scale-free colour, texture and shape descriptors of the segmented plant, gradient boosting) and "
      "CNN-ridge (ResNet18 embedding of the plant crop, ridge or multinomial logistic regression), set as the main model before any "
      "result was seen. '+ DAH' = the same models with the storage days as an additional input. Chance accuracy = 0.25. Full version "
      "with every model and reference in Table S6.", ABBR)

caption("S1", "Capture design from the EXIF metadata of the photographs.")
table(T["S1"])
notes("Each folder (storage time × visual class × genotype) is one capture session of about 100 photographs; the camera dates of T2 and "
      "T3 do not match 6 and 10 DAH, and the folder labels are used.")
caption("S2", "Shelf life of each genotype (Phase 1, online survey, survival analysis) used to define the remaining shelf life.")
table(T["S2"])
notes("t50 = storage time at which half of the respondents would no longer buy the genotype; values above 14 d are truncated at 14 d, "
      "the last observed time. Remaining shelf life = shelf life used − DAH; a photograph is to be rejected when it is ≤ 0.")
caption("S3", "Pseudostems identified in the photographs (no identifier was recorded at capture).")
table(T["S3"])
notes("Contiguous runs = Ward clustering of consecutive photographs within each folder; a pseudostem never spans two folders, so the "
      "session-level validation does not depend on this identification.")
caption("S4", "Image processing: photographs in which the plant was found and properties of the aligned plant, by storage time and visual class.")
table(T["S4"])
notes("The plant is segmented on the table (background modelled by a quadratic surface), cleaned of shadows, reflections, walls and "
      "objects, aligned along its main axis with the base on the left and normalised to a fixed length (no scale reference was available).")
caption("S5", "Correlation over the 40 genotype × time cells between the camera colour of the aligned plant and the colorimeter, before and "
        "after the per-session colour correction.")
table(T["S5"])
notes("Base = left 30 % of the aligned plant (≈ pseudostem); tip = right 30 % (≈ leaves). Correction = per-channel gain in linear RGB that "
      "brings the median colour of the table (neutral reference present in every photograph) of each session to a fixed grey.")
caption("S6", "Validation of every photo model and every metadata reference under the five protocols (full version of Table 1).")
table(T["S6"], widths=[4.3, 3.0, 1.9] + [(USABLE - 9.2) / 5] * 5, group_first=True)
notes("Values with the " + CI + " References: trivial (training mean or class proportions), DAH only, genotype only and genotype + DAH "
      "(remaining shelf life = shelf life of the genotype − DAH, hence R² = 1 when the genotype is known).", ABBR)
caption("S7", "Difference in R² per cell against the DAH-only reference (model − DAH only), with the 95% interval from a paired cluster bootstrap.")
table(T["S7"], group_first=True)
notes("An interval entirely above 0 means that the photograph adds information beyond the storage days.", ABBR)
caption("S8", "Per-photograph R² and class accuracy of the plant-only models and of the previous version.")
table(T["S8"], group_first=True)
notes("CTD-GB (v1): descriptors with the previous segmentation; CNN-MTL (v1): whole photograph including the background. The previous "
      "version was not validated with P4.", ABBR)
caption("S9", "Stability of P1 over five random partitions of the genotype × time blocks (seeds 42–46), ensemble.")
table(T["S9"], decimals=3)
caption("S10", "The same models validated with partitions from ungrouped (random photograph) to unseen genotype, per photograph.")
table(T["S10"])
notes("v1 = CTD-GB with the previous segmentation; v2 = ensemble on the plant alone. The drop from the random-photograph partition to the "
      "session-level partitions is the part of the performance that came from recognising the session or the pseudostem "
      "(Kapoor and Narayanan, 2023; Roberts et al., 2017).")
caption("S11", "Genotype not seen in training (P3): mean absolute error of the sample estimate when k pseudostems of the sample are photographed.")
table(T["S11"])
notes("Ensemble; photographs of each pseudostem averaged; 300 random draws of k pseudostems per genotype × time cell.")
caption("S12", "Accept / reject decisions against the truth (remaining shelf life ≤ 0 = reject), share of photographs in each outcome.")
table(T["S12"].drop(columns=[c for c in T["S12"].columns if c.startswith("_")], errors="ignore"), group_first=True, decimals=1)
notes("Days rule = reject when the storage days exceed the median shelf life of the ten genotypes. '+ conformal' = the photograph "
      "decides only when its 90% split-conformal interval does not cross 0; otherwise the days rule decides. Sensitivity = share of "
      "rejectable photographs that are rejected; 95% CI by cluster bootstrap.", ABBR)
w = T["S13"].copy()
w["Coverage / width"] = [f"{c:.2f} / {x:.1f}" for c, x in zip(w["Coverage (target 0.90)"], w["Mean width"])]
w = w.pivot_table(index=["Output", "Model"], columns="Protocol", values="Coverage / width", aggfunc="first").reset_index()
caption("S13", "Empirical coverage and mean width of the 90% split-conformal prediction intervals on the test photographs.")
table(w, group_first=True)
notes("Calibration on 15 % of the training units, separated by the same unit that each protocol leaves out (Angelopoulos and Bates, "
      "arXiv:2107.07511). Coverage below 0.90 means that the test units differ from the calibration units more than these differ among themselves.")
caption("S14", "Accuracy of models that see only the background of the photographs (outside the plant mask), previous version.")
table(T["S14"])
notes("Background colour = Lab statistics of everything outside the dilated plant mask; environment embedding = ResNet18 embedding of the "
      "photograph with the whole table painted its median colour. Partitions: random photographs, sessions left out, capture days left "
      "out. The background identified the session, and through it the storage time: a shortcut that version 2 removes.")

doc.save(OUT)
print("guardado", OUT)
