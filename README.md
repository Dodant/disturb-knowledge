# DisturbKnowledge
## Experiment Result
### CIFAR-10
Teacher - densenet_bc_k12_depth100
|  Student   |  w/o KD | w/ KD |  ours (noise rate = 10%) |
|----------|----------------|---------------|-------------|
| resnet20 | 92.60 | | **92.71** | 
| resnet32 | 93.53 | | **93.72** |
| resnet44 | 94.01 | | 93.95 |
| resnet56 | 94.37 | | **94.40** |


### CIFAR-100
Teacher - wide_resnet_40_4
|  Student  |  w/o KD | w/ KD |  ours (noise rate = 10%) |
|----------|----------------|---------------|-------------|
| resnet20 | 68.83 | | **69.4** | 
| resnet32 | 70.16 | | **71.25** |
| resnet44 | 71.63 | | **71.79** |
| resnet56 | 72.63 | | **72.79** |
