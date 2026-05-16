# Datasets

This repository uses the four PeMS traffic-flow benchmarks supported by
BasicTS:

| Dataset | Nodes | Time steps | Frequency |
| --- | ---: | ---: | --- |
| PEMS03 | 358 | 26,208 | 5 min |
| PEMS04 | 307 | 16,992 | 5 min |
| PEMS07 | 883 | 28,224 | 5 min |
| PEMS08 | 170 | 17,856 | 5 min |

Metadata files (`desc.json`) are included so configs can read the standard
BasicTS settings. Raw data files are not included. Download or prepare the
datasets in BasicTS format so that each dataset directory contains at least:

```text
datasets/PEMS04/
  data.dat
  adj_mx.pkl
  desc.json
```

Adjacency files are part of the standard BasicTS-ready PeMS datasets. ST-MVD is
graph-free and does not use them.
