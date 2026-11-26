# Protein subcellular localization with protein language models

Predicting where a protein ends up in the cell (10 compartments) from its amino-acid
sequence, on DeepLoc 1.0. The project compares a classical k-mer + XGBoost baseline,
a 1D CNN trained from scratch, frozen ESM-2 embeddings with a small MLP head, and
LoRA fine-tuning of ESM-2.

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
- **ESM-2 + LoRA**: the same ESM-2 with low-rank adapters (r=8, alpha=16) on the attention query and value matrices, mean pooling and the same MLP head, all trained jointly. 310,986 trainable parameters (about 0.9% of the model; 184,320 of them are adapter weights). Learning rate 3e-4 for the adapters and 1e-3 for the head, 4 epochs, bfloat16 autocast, batch size 16.

The CNN and the frozen-embedding model use sequences trimmed to 1,000 residues (first and
last 500 for longer ones). LoRA fine-tuning uses 510 residues (first and last 255) to fit
in 6 GB of GPU memory.

## Results (validation, mean ± std over 3 training seeds, one fixed split)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| k-mer (1-3) + XGBoost | 0.522 ± 0.001 | 0.344 ± 0.003 |
| 1D CNN | 0.614 ± 0.007 | 0.426 ± 0.012 |
| Frozen ESM-2 (35M) + MLP | 0.750 ± 0.004 | 0.655 ± 0.008 |
| ESM-2 (35M) + LoRA | 0.791 ± 0.004 | 0.665 ± 0.005 |

- Pretrained protein embeddings give by far the largest gain over models trained from scratch.
- LoRA improves accuracy by about 4 points over frozen embeddings. The Macro-F1 difference
  (about 1 point) is within the seed-to-seed spread, so it is not established.
- The two rarest classes (lysosome/vacuole and peroxisome, 52 and 25 validation proteins)
  remain the hardest for every model, and their per-class scores are very noisy.

The best epoch of the CNN and of the frozen-embedding head is chosen on the validation set,
so those numbers are slightly optimistic; LoRA uses a fixed 4-epoch schedule and its final
epoch was also its best one (validation Macro-F1 was still rising). The seed spread only
reflects training randomness. Test-set results will be reported once, after all settings
are fixed.

## Repository layout
- `src/`: data loading and splitting (`data.py`), models (`models.py`), training loops (`train.py`, `lora.py`), ESM-2 embedding extraction (`esm_embed.py`)
- `notebooks/`: `01_baseline_and_cnn`, `02_frozen_esm2`, `03_lora_esm2`
- `tests/`: unit tests (`python -m pytest -q`)
- `outputs/`: result tables (CSV)

## Run
`pip install -r requirements.txt`, place the FASTA file in `data/`, then run the notebooks in order.
