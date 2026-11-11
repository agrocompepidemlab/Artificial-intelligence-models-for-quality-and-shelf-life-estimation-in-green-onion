# Data

Everything the three notebooks read is in this folder, so a fresh clone reproduces the same numbers and figures.
The only files left out are those needed to rebuild the Phase 3 caches from the photographs. They are too large for the
repository: the 16,396 photographs, the aligned crops (2.3 GB) and the per-fold files of the validation.

## Phases 1 and 2

| File | Content | Read by |
|---|---|---|
| `Phase_1_Consumer_perception.xlsx` | Consumer survey, online and in person (three sheets, original Spanish names; see the main README) | Phase 1 and Phase 2 notebooks |
| `Phase_2_Physicochemical.xlsx` | Physicochemical traits of 406 pseudostems (sheet `Sheet1`; column `Experimento` = harvest) | Phase 2 and Phase 3 notebooks |
| `Phase_2_GQI.xlsx` | Curated Phase 2 dataset, kinetic analyses and column dictionary | — (supporting data of the article) |
| `Phase_2_Discriminant_bands.xlsx` | Spectral bands that best separate the two species | — (supporting data of the article) |
| `Phase_2_Metabolomics.csv` | Metabolomic profiles | — (supporting data of the article) |

## Phase 3 — `phase3/`

The notebook looks for this folder by default. To use another copy, set the `GREEN_ONION_DATA` variable (for this folder)
or `GREEN_ONION_V2` (for `cache_v2/`).

| File | Content | Produced by |
|---|---|---|
| `dataset_final.csv` | One row per photograph (16,396): original path, storage time, visual class, genotype, DAH, consumer shelf life and the laboratory traits matched to it | Data preparation (never modified) |
| `cache_embeddings_cnn.npz` | ResNet18 embedding of each whole photograph, used to find runs of photographs of the same pseudostem | Earlier whole-photograph analysis |
| `cache_descriptores_mejorados.npz` | Colour and texture descriptors of the whole photograph (CTD-GB v1) | Earlier whole-photograph analysis (`scripts/phase3_validation.py` reads it) |
| `cache/photo_metadata.csv` | Date and time of each photograph (EXIF) | Phase 3 notebook (`RUN_METADATA = True`, needs the photographs) |
| `cache/tramos.csv` | Pseudostem of each photograph (contiguous runs) | Phase 3 notebook (`RUN_TRAMOS = True`) |
| `cache/validation.pkl` | Predictions of the whole-photograph comparison (CTD-GB v1, CNN-MTL v1) under the same protocols | `scripts/phase3_validation.py` |
| `cache/leakage_ladder.pkl` | Leakage ladder of the whole-photograph comparison | `scripts/phase3_leakage_ladder.py` |
| `cache/background_models.pkl` | Accuracy of models that see only the background, plant erased (Table 3a) | `scripts/phase3_background_control.py` |
| `cache_v2/features_v2.npz` | 99 scale-free descriptors and the ResNet18 embedding of the plant alone, per photograph | `scripts/phase3v2_images.py`, `phase3v2_embed.py` |
| `cache_v2/meta_v2.csv` | Segmentation result per photograph (plant found, length, rotation, colour gain of its session) | `scripts/phase3v2_embed.py` |
| `cache_v2/examples_v2.npz` | Example photographs with their mask (Fig. 5A) | `scripts/phase3v2_examples.py` |
| `cache_v2/fig5b_crops.npz` | The 16 aligned crops of Fig. 5B, so that the figure does not need `crops.npy` | Phase 3 notebook, when `crops.npy` is present |
| `cache_v2/validation_v2.pkl.gz` | Predictions of every model and reference under the five protocols, gzip-compressed (the notebook reads it as is) | `scripts/phase3v2_validation.py` |
| `cache_v2/ladder_v2.pkl` | Leakage ladder of the plant-only models | `scripts/phase3v2_validation.py --ladder` |
| `cache_v2/occlusion_v2.npz` | Occlusion maps (Fig. S3) | `scripts/phase3v2_occlusion.py` |
| `cache_v2/tables_v2.pkl` | The three tables of the article and the detail tables D1–D14 | Phase 3 notebook (section 8) |

How to rebuild the caches from the photographs is described in the main README.
