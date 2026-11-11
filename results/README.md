# Results

Everything here is written by the notebooks or by the scripts from the notebooks' output; re-running them gives the same
files. The figures of every phase are shown inside the notebooks and are not written to disk.

| File | Phase | Written by |
|---|---|---|
| `GreenOQ_results.csv` | 2 | GQI per genotype × storage time (`notebooks/Phase_2_GreenOQ.ipynb`) |
| `GQI_classification.csv` | 2 | Mean GQI and quality class per genotype (`notebooks/Phase_2_GreenOQ.ipynb`) |
| `Phase3_tables_supplementary_material_6.docx` | 3 | The three Phase 3 tables of the article, in English with their notes (`scripts/phase3v2_tables_docx.py`, from section 8 of `notebooks/Phase_3_AI_models.ipynb`) |
| `previous_version/` | 3 | Results of the first version of Phase 3 (whole photograph), superseded; kept for transparency |

## Phase 3 in numbers

Ensemble of CTD-GB v2 and CNN-ridge on photographs of the plant alone, validated with a whole genotype left out
(P3, the generalization result). Values with 95% cluster-bootstrap CI; full table in Table 2 of
`Phase3_tables_supplementary_material_6.docx`.

| Output (unit) | Photograph | Reference that uses only the metadata |
|---|---|---|
| Visual class, 4 classes (session) | accuracy 0.56 [0.46, 0.66]; weighted κ 0.57 [0.41, 0.73] | 0.25 (chance; classes balanced by design) |
| Storage age, DAH (genotype × time cell) | R² 0.71 [0.63, 0.78] | — |
| GQI (cell) | R² 0.57 [0.37, 0.73] | DAH only 0.38 [0.12, 0.58]; difference +0.20 [0.02, 0.43] |
| Remaining shelf life (cell) | R² 0.61 [0.52, 0.63]; with the storage days 0.69 [0.59, 0.77] | DAH only 0.59 [0.44, 0.71]; difference +0.02 [−0.12, 0.15] and +0.11 [0.01, 0.21] |

- **Accept / reject (P3):** the photograph together with the storage days lowers false acceptance from 15.5% (days rule) to
  10.7% of the photographs, at the cost of 5.4% false rejection (0% for the days rule).
- **Unseen genotypes and photo days (P4)** are harder: class accuracy 0.40, R² 0.48 for DAH and 0.08 for GQI.
- **Leakage control:** with random photographs instead of whole units the R² of DAH would reach 0.86. In the first version,
  a model that saw only the background reached 0.97 accuracy for the storage time. Both are in Table 3.
