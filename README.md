# Artificial intelligence models for quality and shelf-life estimation in green onion

Analysis notebooks for estimating the quality and shelf life of green onion (*Allium fistulosum* and hybrid) genotypes, combining consumer perception surveys with physicochemical measurements.

## Repository structure

```
├── data/
│   └── Phase_1_Consumer_perception.xlsx   # consumer survey responses
├── notebooks/
│   ├── Phase_1_Consumer_perception.ipynb  # consumer perception and acceptance analysis
│   └── Phase_2_GreenOQ.ipynb              # GreenOQ quality index (GQI)
├── results/
│   ├── GreenOQ_results.csv                # GQI per genotype × shelf-life time
│   └── GQI_classification.csv             # mean GQI and quality class per genotype
└── requirements.txt
```

## Phase 1 — Consumer perception

`notebooks/Phase_1_Consumer_perception.ipynb` analyzes the consumer survey for ten genotypes:

- **a)** Likert (1–5) ratings of visual attributes (pseudostem color, length and diameter; leaf color and length).
- **b)** Purchase intention by genotype (No / Maybe / Yes).
- **c)** Acceptance curve across shelf life derived from the *"when would you stop purchasing"* question.
- **d)** Acceptance curve from direct selection of genotypes at each timepoint (T1–T4).
- **e)** Acceptance heatmap for the balanced panel (respondents who answered T1–T4).

Input: `data/Phase_1_Consumer_perception.xlsx` (sheets `Preguntas por material` and `Preguntas en vida util`).

## Phase 2 — GreenOQ Index (GQI)

`notebooks/Phase_2_GreenOQ.ipynb` builds a 0–100 quality index from five variables:

| Variable | Symbol | Weight source | Utility |
|---|---|---|---|
| Pseudostem length | L | Consumer surveys A + B | increasing min–max |
| Pseudostem diameter | D | Consumer surveys A + B | increasing min–max |
| Pseudostem hue | h | Consumer surveys A + B | Gaussian around fresh (T1) hue |
| Firmness | F | PCA communality | decreasing min–max |
| Pyruvic acid | AP | PCA communality | Gaussian around panel median |

The visible and lab blocks are combined with equal mass (50/50). Genotypes are classified as Low / Medium / High quality by terciles of their mean GQI. The notebook also compares earlier weighting schemes (survey only, 60/40) as a sensitivity check.

Inputs (place them in `data/`; they are not included in the repository):

- `ANALISIS FISICOQUIMICO COMPLETO.xlsx` — sheet `Sheet1`, physicochemical analysis per genotype and time.
- `AcPiruvicoVidaUtil.xlsx` — sheet `AcPiruvico`, pyruvic acid at T1 and T4.

## How to run

```bash
pip install -r requirements.txt
jupyter lab   # or open the notebooks in VS Code
```

The notebooks can be run from the repository root or from `notebooks/`; paths are resolved relative to the repository. Figures use Times New Roman, which must be installed on the system.

Source spreadsheets keep their original Spanish sheet names, column names and answer labels (e.g. `Genotipo`, `Tiempo`, `Tal vez`, `Sí`); the notebooks refer to them as-is.
