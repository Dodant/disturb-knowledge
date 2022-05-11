


# CIFAR10 - Teacher: densenet_bc_k12_depth100

## S: Resnet 20
# python examples/image_classification.py --config configs/sample/cifar10/mix/resnet20_from_densenet_bc_k12_depth100-mix.yaml --log log/cifar10/mix/resnet20_from_densenet_bc_k12_depth100_30.log
## S: Resnet 32
# python examples/image_classification.py --config configs/sample/cifar10/mix/resnet32_from_densenet_bc_k12_depth100-mix.yaml --log log/cifar10/mix/resnet32_from_densenet_bc_k12_depth100_30.log
## S: Resnet 44
# python examples/image_classification.py --config configs/sample/cifar10/mix/resnet44_from_densenet_bc_k12_depth100-mix.yaml --log log/cifar10/mix/resnet44_from_densenet_bc_k12_depth100_30.log
## S: Resnet 56
# python examples/image_classification.py --config configs/sample/cifar10/mix/resnet56_from_densenet_bc_k12_depth100-mix.yaml --log log/cifar10/mix/resnet56_from_densenet_bc_k12_depth100_30.log


# CIFAR100 - Teacher: wide_resnet40_4
# NO KD
python image_classification_yechan.py --config configs/sample/cifar100/ce/resnet20.yaml --log log/cifar100/ce/resnet20.log
python image_classification_yechan.py --config configs/sample/cifar100/ce/resnet32.yaml --log log/cifar100/ce/resnet32.log
python image_classification_yechan.py --config configs/sample/cifar100/ce/resnet44.yaml --log log/cifar100/ce/resnet44.log
python image_classification_yechan.py --config configs/sample/cifar100/ce/resnet56.yaml --log log/cifar100/ce/resnet56.log

# alpha = 10 percent
# w/ KD lambda=0.05
python image_classification_yechan.py --config configs/sample/cifar100/alpha10per/mix/resnet20_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha10per/mix/resnet20_from_wide_resnet40_4_5per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha10per/mix/resnet32_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha10per/mix/resnet32_from_wide_resnet40_4_5per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha10per/mix/resnet44_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha10per/mix/resnet44_from_wide_resnet40_4_5per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha10per/mix/resnet56_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha10per/mix/resnet56_from_wide_resnet40_4_5per.log

# alpha = 20 percent
# w/ KD lambda=0.05
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_5per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_5per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_5per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_5per.yaml --log log/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_5per.log

# w/ KD lambda=0.10
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_10per.yaml --log log/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_10per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_10per.yaml --log log/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_10per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_10per.yaml --log log/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_10per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_10per.yaml --log log/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_10per.log

# w/ KD lambda=0.20
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_20per.yaml --log log/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_20per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_20per.yaml --log log/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_20per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_20per.yaml --log log/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_20per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_20per.yaml --log log/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_20per.log

# w/ KD lambda=0.30
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_30per.yaml --log log/cifar100/alpha20per/mix/resnet20_from_wide_resnet40_4_30per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_30per.yaml --log log/cifar100/alpha20per/mix/resnet32_from_wide_resnet40_4_30per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_30per.yaml --log log/cifar100/alpha20per/mix/resnet44_from_wide_resnet40_4_30per.log
python image_classification_yechan.py --config configs/sample/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_30per.yaml --log log/cifar100/alpha20per/mix/resnet56_from_wide_resnet40_4_30per.log


## S: Resnet 20
# python examples/image_classification.py --config configs/sample/cifar100/mix/resnet20_from_wide_resnet40_4-mix.yaml --log log/cifar100/mix/resnet20_from_wide_resnet40_4_30.log
## S: Resnet 32
# python examples/image_classification.py --config configs/sample/cifar100/mix/resnet32_from_wide_resnet40_4-mix.yaml --log log/cifar100/mix/resnet32_from_wide_resnet40_4_30.log
## S: Resnet 44
# python examples/image_classification.py --config configs/sample/cifar100/mix/resnet44_from_wide_resnet40_4-mix.yaml --log log/cifar100/mix/resnet44_from_wide_resnet40_4_30.log
## S: Resnet 56
# python examples/image_classification.py --config configs/sample/cifar100/mix/resnet56_from_wide_resnet40_4-mix.yaml --log log/cifar100/mix/resnet56_from_wide_resnet40_4_30.log


# # CIFAR10 - Teacher: densenet_bc_k12_depth100 !!! KDLoss replace!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
#
## S: Resnet 20
# python examples/image_classification.py --config configs/sample/cifar10/kd/resnet20_from_densenet_bc_k12_depth100-final_run.yaml --log log/cifar10/kd/resnet20_from_densenet_bc_k12_depth100.log
## S: Resnet 32
# python examples/image_classification.py --config configs/sample/cifar10/kd/resnet32_from_densenet_bc_k12_depth100-final_run.yaml --log log/cifar10/kd/resnet32_from_densenet_bc_k12_depth100.log
## S: Resnet 44
# python examples/image_classification.py --config configs/sample/cifar10/kd/resnet44_from_densenet_bc_k12_depth100-final_run.yaml --log log/cifar10/kd/resnet44_from_densenet_bc_k12_depth100.log
## S: Resnet 56
# python examples/image_classification.py --config configs/sample/cifar10/kd/resnet56_from_densenet_bc_k12_depth100-final_run.yaml --log log/cifar10/kd/resnet56_from_densenet_bc_k12_depth100.log
#
## CIFAR100 - Teacher: wide_resnet40_4
#
## S: Resnet 20
# python examples/image_classification.py --config configs/sample/cifar100/kd/resnet56_from_wide_resnet40_4_10per.yaml --log log/cifar100/kd/resnet20_from_wide_resnet40_4.log
## S: Resnet 32
# python examples/image_classification.py --config configs/sample/cifar100/kd/resnet56_from_wide_resnet40_4_20per.yaml --log log/cifar100/kd/resnet32_from_wide_resnet40_4.log
## S: Resnet 44
# python examples/image_classification.py --config configs/sample/cifar100/kd/resnet56_from_wide_resnet40_4_30per.yaml --log log/cifar100/kd/resnet44_from_wide_resnet40_4.log
## S: Resnet 56
# python examples/image_classification.py --config configs/sample/cifar100/kd/resnet56_from_wide_resnet40_4_10per.yaml --log log/cifar100/kd/resnet56_from_wide_resnet40_4.log