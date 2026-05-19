# ST-MVD

Official minimal code release for **ST-MVD: Multi-View Decomposition for
Spatial-Temporal Forecasting**.

This repository is a lightweight BasicTS-based project. It keeps the BasicTS
training pipeline, ST-MVD model code, and PeMS configs, while removing unrelated
baseline models and configs. The ST-MVD implementation lives in
`baselines/ST_MVD`.

## Layout

```text
baselines/ST_MVD/
  PEMS03.py
  PEMS04.py
  PEMS07.py
  PEMS08.py
  arch/
    stmvd.py
    mlp.py
basicts/
experiments/
  train.py
  evaluate.py
requirements.txt
```

The project is based on the `hotfix/deepar` branch of
[GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS/tree/hotfix/deepar).

## Setup

Install PyTorch for your CUDA/CPU environment first. Then install the remaining
dependencies:

```bash
pip install -r requirements.txt
```

## Data

Download or prepare the PeMS datasets in the standard BasicTS format. Metadata
files are included for the four PeMS benchmarks, but the actual `data.dat`
files are not distributed. Each dataset should follow this layout:

```text
datasets/PEMS04/
  data.dat
  adj_mx.pkl
  desc.json
```

The adjacency file is part of the standard BasicTS-ready PeMS dataset. ST-MVD
does not use graph structure, but keeping the dataset layout unchanged makes the
repository compatible with the BasicTS runner.

## Training

Run from the repository root:

```bash
python experiments/train.py -c baselines/ST_MVD/PEMS04.py -g 0
```

Available configs match the main-paper ST-MVD settings:

| Config | Nodes | Relational views K | Per-view dim | Temporal branches V_t | ToD | DoW |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `baselines/ST_MVD/PEMS03.py` | 358 | 2 | 24 | 4 | Yes | No |
| `baselines/ST_MVD/PEMS04.py` | 307 | 3 | 42 | 4 | Yes | Yes |
| `baselines/ST_MVD/PEMS07.py` | 883 | 3 | 42 | 4 | Yes | Yes |
| `baselines/ST_MVD/PEMS08.py` | 170 | 2 | 24 | 5 | Yes | Yes |

## Reproducibility

The four released configs are the exact main-result configurations. Each config
sets `CFG.ENV = {'SEED': 42, 'DETERMINISTIC': True}` and uses
`fingerprint_seed = 42` for the frozen random orthogonal fingerprints.

Shared training hyperparameters:

| Hyperparameter | Value |
| --- | --- |
| Input / output length | 12 / 12 |
| Epochs | 100 |
| Optimizer | Adam |
| Learning rate | 0.002 |
| Weight decay | 0.0001 |
| LR schedule | MultiStepLR at epochs 1, 20, 40, 60, 80, gamma 0.5 |
| Gradient clipping | max norm 5.0 |
| Train / val / test batch size | 32 / 64 / 64 |
| Input-frequency views | 5 |
| Temporal branch output dim | 16 |
| Decoder depth | 3 residual MLP blocks |

## Model

ST-MVD is a graph-free MLP model that builds explicit views before forecasting:

- frequency views from a network-wide mean signal and frequency-band residuals;
- relational views from frozen orthogonal fingerprints interpreted by shared MLPs;
- temporal views from parallel branches with independent temporal embeddings.

The main model class is `STMVD` in `baselines/ST_MVD/arch/stmvd.py`.

## Acknowledgement

Built on the [BasicTS](https://github.com/GestaltCogTeam/BasicTS) training framework.
