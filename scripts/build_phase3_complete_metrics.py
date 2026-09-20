"""Merge the three Phase 3 workbooks into one organized workbook, keeping the original formatting.

The notebook notebooks/Phase_3_AI_models.ipynb writes three workbooks to results/. This script
joins them into results/Phase3_complete_metrics.xlsx, with an index sheet and the sheets ordered
from the headline results to the supporting data.

    python scripts/build_phase3_complete_metrics.py [source folder] [output file]

Both arguments default to results/.
"""
import sys
from copy import copy
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC / "Phase3_complete_metrics.xlsx"

HYPER = SRC / "Phase3_hyperparameters_and_validation.xlsx"
BYMODEL = SRC / "Phase3_results_by_model.xlsx"
DATASET = SRC / "Phase3_results_dataset.xlsx"

# (source workbook, sheet in it, new name, section, what it holds)
PLAN = [
    (HYPER,   "Hyperparameters",    "01 Hyperparameters",     "Models and validation design", "Final configuration of CTD-GB and CNN-MTL, with the source of every value"),
    (HYPER,   "Validation design",  "02 Validation design",   "Models and validation design", "Unit of analysis, bulb identification and the three protocols"),
    (HYPER,   "Partition sizes",    "03 Partition sizes",     "Models and validation design", "Bulbs and images in each test partition"),
    (HYPER,   "Validation summary", "04 Validation summary",  "Headline results",             "The table to report: every scheme, both models"),
    (BYMODEL, "Metrics by output",  "05 Metrics by output",   "Headline results",             "Shelf life, quality state and GQI, with 95% CI"),
    (BYMODEL, "Table S7",           "06 Table S7",            "Headline results",             "Per-bulb metrics with percentile bootstrap CI"),
    (HYPER,   "Train-Val-Test",     "07 Train-Val-Test",      "Detailed metrics",             "Metrics per bulb on training, validation and test sets"),
    (HYPER,   "Runs P1-P2-P3",      "08 Runs P1-P2-P3",       "Detailed metrics",             "Per-bulb metrics of every run of each protocol"),
    (BYMODEL, "CTD-GB",             "09 CTD-GB",              "Detailed metrics",             "Pooled metrics per protocol, CTD-GB"),
    (BYMODEL, "CNN-MTL",            "10 CNN-MTL",             "Detailed metrics",             "Pooled metrics per protocol, CNN-MTL"),
    (BYMODEL, "CNN-MTL no aux",     "11 CNN-MTL no aux",      "Detailed metrics",             "Pooled metrics per protocol, CNN-MTL without auxiliary heads"),
    (BYMODEL, "Trivial baseline",   "12 Trivial baseline",    "Detailed metrics",             "Pooled metrics per protocol, trivial baseline"),
    (HYPER,   "Repeated CV",        "13 Repeated CV",         "Robustness",                   "Repeated grouped CV and the permutation test"),
    (BYMODEL, "Computational cost", "14 Energy and cost",     "Energy and computational cost", "Training time, inference time and estimated energy, per fold and mean"),
    (DATASET, "results",            "15 All metrics (long)",  "Everything in one table",      "Every metric as one row: output, model, validation, subset, metric, value, CI, n, unit"),
    (BYMODEL, "Figure data",        "16 Figure data",         "Supporting data",              "Numbers behind Figures 5 and 6 and the supplementary figures"),
    (BYMODEL, "Predictions P1",     "17 Predictions P1",      "Supporting data",              "Grouped 5-fold CV, one row per test bulb"),
    (BYMODEL, "Predictions P2",     "18 Predictions P2",      "Supporting data",              "Unseen storage time, one row per test bulb"),
    (BYMODEL, "Predictions P3",     "19 Predictions P3",      "Supporting data",              "Unseen genotype, one row per test bulb"),
]

GREEN, LIGHT = "2F5D3A", "E3EEE4"


def copy_sheet(source, target):
    """Copy values, styles, widths, merges and freeze panes from one sheet to another."""
    for row in source.iter_rows():
        for cell in row:
            new = target.cell(row=cell.row, column=cell.column, value=cell.value)
            if cell.has_style:
                new.font = copy(cell.font)
                new.fill = copy(cell.fill)
                new.border = copy(cell.border)
                new.alignment = copy(cell.alignment)
                new.number_format = cell.number_format
    for rng in source.merged_cells.ranges:
        target.merge_cells(str(rng))
    for letter, dim in source.column_dimensions.items():
        if dim.width:
            target.column_dimensions[letter].width = dim.width
    for idx, dim in source.row_dimensions.items():
        if dim.height:
            target.row_dimensions[idx].height = dim.height
    target.freeze_panes = source.freeze_panes
    target.sheet_view.showGridLines = source.sheet_view.showGridLines


books = {path: load_workbook(path) for path in {p for p, *_ in PLAN}}
out = Workbook()
index = out.active
index.title = "00 Index"

for path, sheet, new_name, _section, _what in PLAN:
    copy_sheet(books[path][sheet], out.create_sheet(new_name))

# ---- index -------------------------------------------------------------------------------
index.sheet_view.showGridLines = False
index["A1"] = "Phase 3 - complete metrics"
index["A1"].font = Font(bold=True, size=15, color=GREEN)
index["A2"] = ("AI models estimating shelf life, quality state and GQI of green onion from one photograph. "
               "Every metric of the study in one workbook, plus the energy and computational cost.")
index["A2"].font = Font(italic=True, size=10)

index["A4"], index["B4"], index["C4"] = "Sheet", "Section", "What it holds"
for col in "ABC":
    cell = index[col + "4"]
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=GREEN)
    cell.alignment = Alignment(vertical="center")

row = 5
last_section = None
for _path, _sheet, new_name, section, what in PLAN:
    if section != last_section:
        index.cell(row=row, column=1, value=section).font = Font(bold=True, color=GREEN)
        for col in range(1, 4):
            index.cell(row=row, column=col).fill = PatternFill("solid", fgColor=LIGHT)
        row += 1
        last_section = section
    link = index.cell(row=row, column=1, value=new_name)
    link.hyperlink = "#'" + new_name + "'!A1"
    link.font = Font(color="1F6FB5", underline="single")
    index.cell(row=row, column=2, value=section).font = Font(size=9, color="666666")
    index.cell(row=row, column=3, value=what)
    row += 1

row += 1
index.cell(row=row, column=1, value="Energy note").font = Font(bold=True, color=GREEN)
index.cell(row=row, column=3, value=("Energy = time x 15 W (nominal TDP of the AMD Ryzen 5 3500U); estimated, not metered. "
                                     "Inference excludes feature extraction. See sheet 14."))
row += 2
index.cell(row=row, column=1, value="Built from").font = Font(bold=True, color=GREEN)
index.cell(row=row, column=3, value=("Phase3_hyperparameters_and_validation.xlsx, Phase3_results_by_model.xlsx and "
                                     "Phase3_results_dataset.xlsx, written by notebooks/Phase_3_AI_models.ipynb"))

for letter, width in (("A", 26), ("B", 30), ("C", 105)):
    index.column_dimensions[letter].width = width

for worksheet in out.worksheets:
    worksheet.sheet_view.topLeftCell = "A1"

OUT.parent.mkdir(parents=True, exist_ok=True)
out.save(OUT)
print("saved {}  ({:.0f} KB, {} sheets)".format(OUT, OUT.stat().st_size / 1024, len(out.worksheets)))
