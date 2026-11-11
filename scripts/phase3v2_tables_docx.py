"""Phase 3 v2 · the three tables of Supplementary material 6 as a Word file (English, three-line format, caption above,
notes below).

The tables and every number in the notes come from `tables_v2.pkl`, which the notebook (section 8) saves:
SM6_1, SM6_2, SM6_3A and SM6_3B are the three tables; D1–D14 are the detail tables whose values the notes quote.

Table 1  Samples, labels and photographs per genotype            (detail tables D1–D5)
Table 2  Validation of the photo models under the five protocols  (D6, D7, D9)
Table 3  Controls of the validation and use of the estimate       (D10–D14)

Usage (from the repository root):
    python scripts/phase3v2_tables_docx.py [TABLES.pkl] [OUT.docx]
Defaults: data/phase3/cache_v2/tables_v2.pkl → results/Phase3_tables_supplementary_material_6.docx
"""
import sys, pickle
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "phase3" / "cache_v2" / "tables_v2.pkl"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "Phase3_tables_supplementary_material_6.docx"
with open(SRC, "rb") as fh:
    T = pickle.load(fh)
FONT, SIZE, MINUS = "Times New Roman", 9, "−"

doc = Document()
st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(SIZE)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); st.paragraph_format.space_after = Pt(0)
s0 = doc.sections[0]
s0.orientation = WD_ORIENT.LANDSCAPE; s0.page_width, s0.page_height = Cm(29.7), Cm(21.0)
s0.left_margin = s0.right_margin = Cm(1.8)
s0.top_margin = s0.bottom_margin = Cm(1.3)
USABLE = 29.7 - 3.6


def neg(s):
    """True minus sign in every negative number of a string."""
    s = str(s); out = []
    for i, ch in enumerate(s):
        nxt = s[i + 1] if i + 1 < len(s) else ""
        prev = s[i - 1] if i else " "
        out.append(MINUS if ch == "-" and (nxt.isdigit() or nxt == ".") and not prev.isalnum() else ch)
    return "".join(out)


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


def caption(num, text, new_page=False):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(0 if new_page else 14); p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.page_break_before = new_page; p.paragraph_format.keep_with_next = True
    r = p.add_run(f"Table {num}. "); r.bold = True; p.add_run(text)


def part(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text); r.bold = True; r.italic = True


def notes(*lines):
    for i, l in enumerate(lines):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(3 if i == 0 else 0)
        r = p.add_run(l); r.font.size = Pt(SIZE - 1)


def table(df, widths, group_cols=0, bold_last=False):
    """Three-line table; the first `group_cols` columns are written only when they change."""
    header = [str(c) for c in df.columns]
    rows = [[neg(v) for v in r] for r in df.astype(str).itertuples(index=False)]
    for g in range(group_cols):
        last = None
        for r in rows:
            key = tuple(r[:g + 1])
            if key == last:
                r[g] = ""
            else:
                last = key
    n = len(header); left = group_cols + 1 if group_cols else 1
    assert len(widths) == n and sum(widths) <= USABLE + 0.05, (len(widths), n, sum(widths))
    t = doc.add_table(rows=1 + len(rows), cols=n); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j])
            p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j < left else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val); r.font.size = Pt(SIZE); r.bold = i == 0 or (bold_last and i == len(rows))
            borders(c, top=12 if i == 0 else (6 if i == 1 else None), bottom=12 if i == len(rows) else None)
        if i == 0:
            t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        else:
            t.rows[i]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


ABBR = ("DAH = days after harvest (storage age of the sample); GQI = global quality index (Phase 2); P1 = genotype × time blocks "
        "(sessions) left out; P1 · capture day = capture days left out; P2 = storage time left out; P3 = genotype left out; P4 = one "
        "of the two photo groups left out (five genotypes photographed on their own days), i.e. unseen genotypes and photo days.")
CI = ("95% confidence interval from a cluster bootstrap (2000 resamples of genotype × time blocks in P1, P1 · capture day and P2; "
      "of genotypes in P3); P4 has two photo groups and no interval.")

title = doc.add_paragraph(); r = title.add_run("Supplementary material 6 — Phase 3 tables"); r.bold = True; r.font.size = Pt(12)

# ====================================================================================== Table 1
SM1, D1, D3, D4, D5 = T["SM6_1"], T["D1"], T["D3"], T["D4"], T["D5"]
caption("1", "Samples, labels and photographs of each genotype: photo group, consumer shelf life used to define the remaining shelf "
        "life, and the photographs and pseudostems that enter the models.")
table(SM1, [3.0, 2.2, 5.2, 2.9, 2.9, 2.2, 4.2, 3.5], bold_last=True)


def _days(x):
    d = list(pd.to_datetime(x["Capture date (EXIF)"]).dt.strftime("%d %B").str.lstrip("0"))
    return ", ".join(d[:-1]) + " and " + d[-1]


dates = {g: _days(x) for g, x in D1.sort_values(["Photo group", "Time"]).groupby("Photo group")}
prev, cur = D3.iloc[0], D3.iloc[1]
n_prev = int(sum(prev[c] for c in D3.columns if c.endswith("DAH")))
r5 = {r["Attribute (camera region)"]: (r["r, raw camera"], r["r, corrected camera"]) for _, r in D5.iterrows()}
tot = SM1.iloc[-1]["Photographs (valid)"]
n_all, n_valid = tot.split(" (")[0], tot.split("(")[1].rstrip(")")
notes("Photo groups: five genotypes each, photographed on their own days, group A on " + dates["A"] + " 2026 and group B on " + dates["B"] +
      " 2026 (camera clock). The storage times are labelled 2, 6, 10 and 14 DAH from the folders; the camera dates of the second and "
      "third times do not match 6 and 10 DAH.",
      "Session = one folder (storage time × visual class × genotype) of about 100 photographs of the same sample, taken in a single "
      "day; 4 storage times × 4 visual classes = 16 sessions per genotype, balanced by design.",
      "t50 = storage time at which half of the respondents would no longer buy the genotype (Phase 1, online survey, survival analysis); "
      "values above 14 d are truncated at 14 d, the last observed time. Remaining shelf life = shelf life used − DAH; a sample is to be "
      "rejected when it is ≤ 0, i.e. from the storage time in the column 'Rejected from'.",
      f"Valid = plant segmented and passing the length check ({n_valid} of {n_all}; ≥ {D4['Plant found (%)'].min():.1f}% in every "
      "storage time × visual class). The plant is segmented on the table (background modelled by a quadratic surface), cleaned of "
      "shadows, reflections, walls and objects, aligned along its main axis with the base on the left and normalised in length (no scale "
      "reference was available). The colour of each session is corrected with a per-channel gain in linear RGB (0.76–1.56) that brings "
      "the median colour of the table, a neutral surface present in every photograph, to a fixed grey.",
      "Camera against colorimeter over the 40 genotype × time cells, r (raw / corrected camera): pseudostem L* "
      f"{r5['Pseudostem L* (base)'][0]:.2f} / {r5['Pseudostem L* (base)'][1]:.2f}, pseudostem hue {r5['Pseudostem hue (base)'][0]:.2f} / "
      f"{r5['Pseudostem hue (base)'][1]:.2f}, leaf L* {r5['Leaf L* (tip)'][0]:.2f} / {r5['Leaf L* (tip)'][1]:.2f}, leaf hue "
      f"{r5['Leaf hue (tip)'][0]:.2f} / {r5['Leaf hue (tip)'][1]:.2f} (base = left 30% of the aligned plant; tip = right 30%).",
      "Pseudostems = contiguous runs of photographs (Ward clustering of consecutive photographs within each session; "
      f"{cur['Per session, median (range)']} per session, median (range)); no pseudostem spans two sessions, so the session-level "
      f"validation does not depend on this identification. The previous version counted {n_prev:,} units, "
      f"{prev['Single-photo units at 2 DAH (%)']:.0f}% of them single photographs at 2 DAH.")

# ====================================================================================== Table 2
caption("2", "Validation of the photo models (plant only) and of the metadata references with sessions (P1), capture days, storage "
        "times (P2), genotypes (P3) and groups of genotypes photographed on other days (P4) left out.", new_page=True)
w = [3.9, 4.5, 1.9]; table(T["SM6_2"], w + [(USABLE - sum(w)) / 5] * 5, group_cols=1)
D9 = T["D9"]
sd = dict(zip(D9.iloc[:, 0].str.split(r" \(| —").str[0], D9["SD"]))
notes("Values with the " + CI + " The visual class is evaluated per session (mean class probabilities of its photographs, 160 sessions; "
      "chance accuracy 0.25, chance κ 0) and the other outputs per genotype × time cell (mean prediction of its photographs, 40 cells).",
      "CTD-GB v2 = gradient boosting on 99 scale-free colour, profile, shape and texture descriptors of the segmented plant; CNN-ridge = "
      "ridge or multinomial logistic regression on the ResNet18 (ImageNet, frozen) embedding of the plant crop; Ensemble = mean of the "
      "two, set as the main model before any result was seen; '+ DAH' = the same model with the storage days as an additional input.",
      "References that use only the metadata: DAH only; genotype + DAH (remaining shelf life = shelf life of the genotype − DAH, hence "
      "R² = 1 when the genotype is known; in P3 and P4 the genotype is unseen and it equals DAH only). For the visual class every "
      "metadata reference stays at accuracy 0.25 and κ 0 (classes balanced by design within each genotype and time); the trivial and "
      "genotype-only references have R² ≤ 0.06 for every output and protocol.",
      "ΔR² = paired difference with the DAH-only reference in the same bootstrap resamples; an interval entirely above 0 means that the "
      "photograph adds information beyond the storage days.",
      f"P1 is shown for one random partition (seed 42); over five partitions (seeds 42–46) the SD of the ensemble was "
      f"{sd['Days after harvest']:.3f} (DAH), {sd['GQI']:.3f} (GQI) and {sd['Remaining shelf life']:.3f} (remaining shelf life) in R², and "
      f"{sd['Visual class']:.3f} in class accuracy.", ABBR)

# ====================================================================================== Table 3
caption("3", "Controls of the validation and use of the estimate: (a) leakage ladder and background-only control; (b) accept / reject "
        "decision and 90% prediction intervals under each protocol.", new_page=True)
part("(a) Per-photograph performance when the unit left out of training goes from a random photograph to a whole genotype")
table(T["SM6_3A"], [4.5] + [(USABLE - 4.5) / 9] * 9)
part("(b) Accept / reject decision and 90% split-conformal prediction intervals of the ensemble")
w = [5.6, 5.1]; table(T["SM6_3B"], w + [(USABLE - sum(w)) / 5] * 5, group_cols=1)

D11, D12, D14 = T["D11"], T["D12"], T["D14"]
s3 = D12.set_index(["Protocol", "Decision rule"]).loc["P3 · unseen genotype"]
k = D11.set_index(D11.columns[0])
emb = D14[D14["Background features"].str.startswith("ResNet")].set_index("Target").loc["Storage time"]
notes("(a) v1 = CTD-GB with the previous segmentation; v2 = ensemble on the plant alone; same partitions for both. The drop from the "
      "random-photograph partition to the session-level partitions is the part of the performance that came from recognising the "
      "session or the pseudostem rather than the sample (Roberts et al., 2017; Kapoor and Narayanan, 2023). Background only (v1) = "
      "accuracy of a model that sees only the Lab colour statistics of the photograph outside the dilated plant mask (chance 0.25 for "
      "storage time and class, 0.10 for genotype); with a ResNet18 embedding of the scene without the plant, storage-time accuracy was "
      + ", ".join(f"{emb[c]:.2f}" for c in ["Random photographs", "Session left out", "Capture day left out"]) +
      " with the same three partitions. The background identified the session and, through it, the storage time: the shortcut that "
      "version 2 removes.",
      "(b) Truth: reject when the remaining shelf life is ≤ 0 (40.2% of the photographs). False acceptance = accepted although it should "
      "be rejected; false rejection = rejected although acceptable. Days rule = reject when DAH > 10.7 d (median shelf life of the ten "
      "genotypes); Ensemble = reject when its estimate of the remaining shelf life is ≤ 0; '+ conformal' = the photograph decides only "
      "when its 90% interval does not cross 0, otherwise the days rule decides. In P3, sensitivity was "
      f"{s3.loc['Days rule', 'Sensitivity [95% CI]']} for the days rule and {s3.loc['Ensemble + DAH', 'Sensitivity [95% CI]']} for "
      f"the ensemble + DAH, and specificity {s3.loc['Days rule', 'Specificity [95% CI]']} and {s3.loc['Ensemble + DAH', 'Specificity [95% CI]']} "
      "(95% CI by cluster bootstrap).",
      "Intervals calibrated on 15% of the training units, separated by the same unit that each protocol leaves out (Angelopoulos and "
      "Bates, arXiv:2107.07511); coverage below 0.90 means that the test units differ from the calibration units more than these "
      "differ among themselves.",
      "Sampling (P3, ensemble, photographs of each pseudostem averaged, 300 random draws per cell): the mean absolute error of the "
      f"sample estimate was {k.loc[1, 'MAE Days after harvest (d)']:.2f}, {k.loc[5, 'MAE Days after harvest (d)']:.2f} and "
      f"{k.loc[16, 'MAE Days after harvest (d)']:.2f} d of DAH and {k.loc[1, 'MAE GQI']:.2f}, {k.loc[5, 'MAE GQI']:.2f} and "
      f"{k.loc[16, 'MAE GQI']:.2f} GQI points with 1, 5 and 16 pseudostems photographed.", ABBR)

OUT.parent.mkdir(parents=True, exist_ok=True)
# same bytes on every run: fixed dates in the document properties and in the zip entries
import io, zipfile, datetime
FIXED = datetime.datetime(2026, 11, 10)
doc.core_properties.created = doc.core_properties.modified = FIXED
doc.core_properties.revision = 1
buf = io.BytesIO(); doc.save(buf)
src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as dst:
    for info in src.infolist():
        dst.writestr(zipfile.ZipInfo(info.filename, date_time=FIXED.timetuple()[:6]), src.read(info.filename), zipfile.ZIP_DEFLATED)
print("saved", OUT)
