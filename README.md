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

Available configs:

| Config | Nodes | Relational views K | Temporal branches V_t | Day-of-week embedding |
| --- | ---: | ---: | ---: | --- |
| `baselines/ST_MVD/PEMS03.py` | 358 | 2 | 4 | No |
| `baselines/ST_MVD/PEMS04.py` | 307 | 3 | 4 | Yes |
| `baselines/ST_MVD/PEMS07.py` | 883 | 3 | 4 | Yes |
| `baselines/ST_MVD/PEMS08.py` | 170 | 2 | 4 | Yes |

## Model

ST-MVD is a graph-free MLP model that builds explicit views before forecasting:

- frequency views from a network-wide mean signal and frequency-band residuals;
- relational views from frozen orthogonal fingerprints interpreted by small MLPs;
- temporal views from parallel branches with independent temporal embeddings.

The main model class is `STMVD` in `baselines/ST_MVD/arch/stmvd.py`.

## Acknowledgement

This codebase uses the BasicTS training framework. If this repository is useful
for your work, please also cite BasicTS:

```bibtex
@article{shao2024exploring,
  title={Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis},
  author={Shao, Zezhi and Wang, Fei and Xu, Yongjun and Wei, Wei and Yu, Chengqing and Zhang, Zhao and Yao, Di and Sun, Tao and Jin, Guangyin and Cao, Xin and others},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  volume={37},
  number={1},
  pages={291-305},
  publisher={IEEE}
}
```
