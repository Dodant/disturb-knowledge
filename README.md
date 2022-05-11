# DisturbKnowledge

## Experiment Result
### CIFAR-100
Teacher - wide_resnet_40_4
alpha = 0.1
|  Student  |  w/o KD | w/ KD |  ours (noise rate = 5 / 10 / 20 / 30%) |
|----------|----------------|---------------|-------------|
| resnet20 | 68.74 | 68.90 | 69.06 / **69.40** / 69.25 / 68.82 | 
| resnet32 | 70.09 | 71.22 | 70.57 / 71.25 / **71.26** / 71.05 |
| resnet44 | 72.01 | 71.92 | 71.98 / 71.79 / **72.50** / 72.46 |
| resnet56 | 72.13 | 72.73 | **73.12** / 72.79 / 72.85 / 72.99 |

alpha = 0.2

|  Student  |  w/o KD | w/ KD |  ours (noise rate = 5 / 10 / 20 / 30%) |
|----------|----------------|---------------|-------------|
| resnet20 | 68.74 | **69.49** | 69.09 / 68.81 / 69.23 / 68.78 |
| resnet32 | 70.09 | 71.14 | 71.55 / **71.79** / 70.87 / 70.86 |
| resnet44 | 72.01 | **72.75** | 72.60 / 72.47 / 72.45 / 72.06 |
| resnet56 | 72.13 | 73.40 | 73.16 / 73.02 / **73.46** / 72.71 |

### CIFAR-100 with Label Corruption
symmetric noise
|  Label Corrupt.  |  Teacher |w/o KD | w/ KD |  ours (noise rate = 10 / 20 / 30%) |
|----------|----------------|---------------|-------------|----------|
|20% | 70.14 | 64.67 | 66.03 | 65.72 / 65.82 / **66.19** |
|40% | 64.40 | 56.90 | 58.93 | 58.55 / 58.27 / **58.97** |
|60% | 55.74 | 47.10 | 48.54 | 46.31 / **49.02** / 48.76 |
pairflip
|  Label Corrupt.  |  Teacher |w/o KD | w/ KD |  ours (noise rate = 10 / 20 / 30%) |
|----------|----------------|---------------|-------------|----------|
|20% | 72.01 | 67.94 | **69.21** | 69.16 / 68.80 / 68.80 |
|40% | 57.36 | 51.83 | 55.49 | 54.71 / 55.80 / **56.24** |

## Reference
https://github.com/yoshitomo-matsubara/torchdistill
https://github.com/UCSC-REAL/cores
### Paper
- Learning From Noisy Labels With Deep Neural Networks: A Survey
- Learning With Instance-Dependent Label Noise: A Sample Sieve Approach
