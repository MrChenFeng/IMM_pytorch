# IMM_pytorch
Pytorch implementation of "Unsupervised Learning of Object Landmarks through Conditional Image Generation", Tomas Jakab*, Ankush Gupta*, Hakan Bilen, Andrea Vedaldi, Advances in Neural Information Processing Systems (NeurIPS) 2018.

## Usage
`python train.py -c Configs.yaml [-epochs 10 --data_root ./images]`
Parameters can be set in yaml or one by one. To check all configurable params:
`python train.py -h`

## Dataset
Currently, AFLW and CelebA has been tested.
The path downloaded dataset containing all images should be set as data_root.


