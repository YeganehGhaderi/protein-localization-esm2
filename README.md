# Protein subcellular localization with protein language models

Predicting where a protein ends up in the cell (10 compartments) from its amino-acid
sequence, on DeepLoc 1.0. The project compares a classical k-mer + XGBoost baseline,
a 1D CNN trained from scratch, frozen ESM-2 embeddings with a small MLP head, and
(in progress) LoRA fine-tuning of ESM-2.

## Data
DeepLoc 1.0 training/test FASTA from the DTU Health Tech DeepLoc-1.0 page, saved as
`data/deeploc_data.fasta` (not included in this repository). The 146 proteins annotated
with two locations (Cytoplasm-Nucleus) are dropped, leaving 10 single-label classes.

## Split
The official test set (2,773 proteins, marked `test` in the file) is held out and used
only for the final evaluation. The remaining 11,085 proteins are split ~80/20 into train
(8,867) and validation (2,218), keeping similar sequences together: two sequences are
linked when they share many 5-mers, and each connected group goes entirely to one side.
This is an approximate homology filter that only catches fairly similar sequences, so
validation scores can be slightly optimistic.

## Models
- **k-mer + XGBoost**: relative frequencies of all 1-, 2- and 3-mers.
- **1D CNN**: embedding, parallel convolutions (kernel sizes 3, 9, 15), masked global max pooling, linear classifier.
- **Frozen ESM-2 + MLP**: mean-pooled embeddings from `esm2_t12_35M_UR50D` (35M parameters, weights not updated), followed by a small MLP trained on top.

All models see sequences trimmed to 1,000 residues (first and last 500 for longer ones).

## Results (validation, mean ± std over 3 training seeds, one fixed split)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| k-mer (1-3) + XGBoost | 0.522 ± 0.001 | 0.344 ± 0.003 |
| 1D CNN | 0.614 ± 0.007 | 0.426 ± 0.012 |
| Frozen ESM-2 (35M) + MLP | 0.750 ± 0.004 | 0.655 ± 0.008 |

The best epoch of each run is chosen on the validation set, so these numbers are slightly
optimistic for every model alike. The seed spread only reflects training randomness. Test-set
results will be reported once, after all settings are fixed.

## Run
`pip install -r requirements.txt`, then `python -m pytest -q` and the notebooks in `notebooks/`.
