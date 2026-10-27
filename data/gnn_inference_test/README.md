# GNN Inference Test Files

Drop `.parquet` test samples here for `models/GNN/GNN_inference_run.ipynb`.

Each file must use the same format as the training data
([Spectrum-Cartography-256x256-UCF-50K](https://huggingface.co/datasets/KR-init/Spectrum-Cartography-256x256-UCF-50K)):

| Column          | Contents                                 |
|-----------------|------------------------------------------|
| `building_mask` | flat 256*256 array                       |
| `tx_origin`     | flat 256*256 array (TX location)         |
| `path_loss`     | flat 256*256 array, ground truth in dB   |
| `sparse_samples` (optional) | flat 256*256 array of real measurements in dB, 0 = unmeasured |

If `sparse_samples` is missing, the notebook simulates 1-10% sparse measurements
from `path_loss` (same as training) and scores the reconstruction against it.
Use files from the held-out `test` split so the model has not seen them.
