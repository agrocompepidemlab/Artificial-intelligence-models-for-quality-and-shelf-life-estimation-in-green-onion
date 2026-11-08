# Artificial intelligence models for quality and shelf-life estimation in green onion

Analysis notebooks for estimating the quality and shelf life of green onion (*Allium fistulosum* and hybrid) genotypes, combining consumer perception surveys with physicochemical measurements.

![Graphical abstract](docs/graphical_abstract.png)

## Summary

Artificial intelligence models estimate the quality and shelf life of green onions from multidimensional indicators and RGB images. The work is organized in three phases:

1. **Data acquisition.** A field and virtual survey collects consumer perception of green onion shelf life, and the samples are photographed according to their quality characteristics.
2. **Quality modeling.** An empirical model, the Green Onion Quality Index (GQI), scores green onions on the attributes that consumers rated as important: pseudostem length (*L*) and diameter (*D*) as increasing min–max utilities, pseudostem hue (*h*) as a Gaussian utility, leaf hue (*h<sub>l</sub>*) as a one-sided Gaussian utility (only yellowing is penalised) and firmness (*F*) as a decreasing min–max utility, with fixed declared bounds and weights equal to the consumer importance of each attribute:

   $$\text{GQI}_{gt} = 100 \times \left[0.197\,u^{+}(L) + 0.191\,u^{+}(D) + 0.198\,e^{-\frac{(h-h^{*})^2}{2\sigma_h^2}} + 0.193\,G^{-}(h_l) + 0.222\,u^{-}(F)\right]$$

   The resulting green onion quality model sorts samples into bad, medium and good quality.
3. **Shelf-life modeling.** Two AI approaches trained on the photographs are compared for determining quality and shelf life: color and texture descriptors with gradient boosting (**CTD-GB**), and CNN embeddings with a multitask neural network (**CNN-MTL**). The result is the green onion shelf-life model.

## Repository structure

```
├── docs/
│   └── graphical_abstract.png             # graphical abstract (figure above)
├── data/
│   ├── Phase_1_Consumer_perception.xlsx   # consumer survey responses
│   ├── Phase_2_Physicochemical.xlsx       # physicochemical traits of 406 pseudostems (GQI input)
│   └── Phase_2_GQI.xlsx                   # curated Phase 2 dataset and column dictionary
├── notebooks/
│   ├── Phase_1_Consumer_perception.ipynb  # consumer perception and acceptance analysis
│   ├── Phase_2_GreenOQ.ipynb              # GreenOQ quality index (GQI)
│   └── Phase_3_AI_models.ipynb            # AI models from one photograph (CTD-GB and CNN-MTL)
├── results/
│   ├── README.md                          # the headline result, the metrics and the energy summary
│   ├── figures/                           # the main result figure
│   ├── Phase3_complete_metrics.xlsx       # every Phase 3 metric in one workbook (20 sheets)
│   ├── GreenOQ_results.csv                # GQI per genotype × shelf-life time
│   └── GQI_classification.csv             # mean GQI and quality class per genotype
├── scripts/
│   └── build_phase3_complete_metrics.py   # merges the three Phase 3 workbooks into one
└── requirements.txt
```

The large Phase 3 inputs (photographs, feature caches, stored validation) are not in the repository; see the Phase 3 section below.

**[→ Read the results](results/README.md)** — the headline result, the metrics of both models under the three validation protocols, and the energy cost of training and inference.

## Phase 1 — Consumer perception

`notebooks/Phase_1_Consumer_perception.ipynb` analyzes the consumer survey for ten genotypes in two modalities, always analyzed separately: **in person** (98 consumers who evaluated real samples stored 2, 6, 10 or 14 days after harvest, DAH; one independent group per time) and **online** (59 respondents who saw photographs of each genotype at the four times). The notebook follows the order of the manuscript figures:

| Figure | Analysis |
|---|---|
| 1A | Likert (1–5) ratings of five visual attributes (online, n = 59) |
| 1B | Genotype selections retained relative to 2 DAH (balanced online panel, n = 39); binomial GEE with exchangeable correlation, Bonferroni, cluster-bootstrap 95% CI |
| 1C, 1D | Sensory shelf life by **survival analysis with interval censoring** (Hough et al., 2003): survival curves per genotype, in person (1C) and online (1D); Weibull or log-normal by AIC, Turnbull estimate |
| 1E | Stated purchase decision per genotype (No / Maybe / Yes, online, n = 59) |
| 1F | t50 (day at which 50% of consumers reject) per genotype and modality with 95% bootstrap CI |
| 2A–2T | Per genotype: word cloud of the open answers (analyzed in Spanish, manual glossary to English, negation marked, terms in ≥ 2 answers) and survival curve with censored observations and 95% CI band |
| S1a, S1b | n per modality after quality filters, and origin of the online respondents |

Quality filters (straight-lining, IQR outliers, low agreement with the majority) leave 95 in-person consumers and an online balanced panel of 39. Every random step uses a fixed seed, so the notebook reproduces the same numbers and figures on every run. Figures are shown only inside the notebook (nothing is written to disk).

Input: `data/Phase_1_Consumer_perception.xlsx`, with three sheets (original Spanish names):

| Sheet | Modality | Content |
|---|---|---|
| `Preguntas por material` | online | origin, Likert ratings, and per genotype: *would you buy it?*, *why?* (open text) and *when would you stop buying it?* |
| `Preguntas en vida util` | online | genotypes the respondent would buy at each time T1–T4 (2, 6, 10, 14 DAH), as material numbers 1–10 |
| `Vida util Genotipos` | in person | storage time evaluated and, per genotype, *would you buy it?* (yes/no) with the real sample |

## Phase 2 — Green Onion Quality Index (GQI)

`notebooks/Phase_2_GreenOQ.ipynb` builds a 0–100 index of the **commercial quality** of green onion (commercial size + freshness), following the ten steps of the OECD/JRC (2008) handbook on composite indicators; each step is a section of the notebook.

| Variable | Symbol | Unit | Utility | Fixed parameters | Weight |
|---|---|---|---|---|---|
| Pseudostem length | L | cm | increasing min–max | 15–30 cm | 19.7 % |
| Pseudostem diameter | D | mm | increasing min–max | 5–23 mm | 19.1 % |
| Pseudostem hue | h | ° | Gaussian | h* = 102.9°, σ = 16.0° | 19.8 % |
| Leaf hue | h<sub>l</sub> | ° | one-sided Gaussian | h<sub>l</sub>* = 124.1°, σ = 5.2° | 19.3 % |
| Firmness (puncture force) | F | N | decreasing min–max | 3–30 N | 22.2 % |

- **Variable selection**: consumer importance ≥ 50 %, measured at the four storage times and significant change between genotypes or over storage. Leaf length (29 % importance) is excluded.
- **Weights**: consumer importance from the published survey (A) and the Phase 1 survey (B). PCA is used only to check the structure of the variables, not to weight them.
- **Fixed bounds**: adding or removing genotypes does not change the GQI of the others.
- **Robustness**: alternative weights, leaving one variable out, observed bounds, geometric aggregation and 10 000 random weight vectors.
- **External validation**: linear mixed model (genotype as random effect), ROC against the consumer shelf life of each genotype compared with storage time alone, and correlation with declared (online) and observed (in-person) consumer acceptance, with bootstrap over genotypes.

The pseudostem diameter is labelled "(cm)" in the source sheet, but its values (5.0–23.0) are only plausible in mm; the notebook treats it as mm.

Inputs (in `data/`): `Phase_2_Physicochemical.xlsx` (sheet `Sheet1`, 406 pseudostems) and `Phase_1_Consumer_perception.xlsx` (validation). Outputs (in `results/`): `GreenOQ_results.csv` and `GQI_classification.csv`.

> The Phase 3 models were trained on the previous version of the GQI (with pyruvic acid and PCA weights); they will be retrained on this version.

## Phase 3 — AI models from one photograph

`notebooks/Phase_3_AI_models.ipynb` estimates the remaining shelf life, the quality state and the GQI of Phase 2 from a single RGB photograph of the pseudostem. It is self-contained: all the code lives in the notebook.

Two models are compared, both trained on 16,396 photographs:

- **CTD-GB** — 109 color, texture and shape descriptors measured on the segmented pseudostem, with histogram gradient boosting (simple and interpretable).
- **CNN-MTL** — those descriptors plus 512 ResNet18 embeddings, with a multitask neural network that predicts shelf life, quality state and GQI at once, using the 61 laboratory traits as auxiliary targets.

Validation is bulb-grouped so no bulb appears in both training and test: 5-fold cross-validation (P1), unseen storage time (P2) and unseen genotype (P3), plus repeated validation and a permutation test for leakage control.

### Data

Place these files in `data/`, or point the environment variable `GREEN_ONION_DATA` at the folder that holds them. They are not in the repository because of their size:

| File | What it is |
|---|---|
| `dataset_final.csv` | 16,396 photographs with genotype, storage time, quality state, shelf life and 61 laboratory traits |
| `cache_descriptores_mejorados.npz` | the 109 descriptors already measured on every photograph |
| `cache_embeddings_cnn.npz` | the 512 ResNet18 embeddings of every photograph |
| `GQI_vs_modelo_completo_datos.xlsx` | GQI of Phase 2, per genotype × storage time |
| `cache/predictions.pkl`, `cache/extra_validation.pkl`, `cache/cnn_mtl_demo.pt` | stored validation output and the demo model, to skip the long runs |

`Phase_1_Consumer_perception.xlsx` is already in `data/`. The photographs themselves are not needed to reproduce any number — the two caches hold everything the models read. They are only required to rebuild those caches or to display photographs; if you have them, set `GREEN_ONION_PHOTOS` to the folder containing `GreenOnionCap3`.

### Running it

The three flags at the top of the notebook control how much is recomputed. With all of them `False` (the default) the notebook loads the stored results and runs in about 20 minutes:

| Flag | `True` recomputes | Time |
|---|---|---|
| `RUN_FEATURE_EXTRACTION` | the two image-feature caches (needs the photographs) | several hours |
| `RUN_FULL_VALIDATION` | P1, P2, P3 and the grouped split | ~50 min |
| `RUN_EXTRA_VALIDATION` | repeated validation and the permutation test | ~30 min |

Outputs go to `results/`: figures at 300 dpi in `results/Phase_3_figures/` (PNG and PDF), and the workbooks `Phase3_hyperparameters_and_validation.xlsx`, `Phase3_results_by_model.xlsx` and `Phase3_results_dataset.csv`/`.xlsx`.

## How to run

```bash
pip install -r requirements.txt
jupyter lab   # or open the notebooks in VS Code
```

The notebooks can be run from the repository root or from `notebooks/`; paths are resolved relative to the repository. Figures use Times New Roman, which must be installed on the system.

Source spreadsheets keep their original Spanish sheet names, column names and answer labels (e.g. `Genotipo`, `Tiempo`, `Tal vez`, `Sí`); the notebooks refer to them as-is.
