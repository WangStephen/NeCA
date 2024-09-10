# NeCA: 3D Coronary Artery Tree Reconstruction from Two 2D Projections by Neural Implicit Representation

# 1. Overview

This is the code repository for the [NeCA](https://arxiv.org/abs/2409.04596) paper by Yiying Wang, Abhirup Banerjee and Vicente Grau.

## Citation

If you find the code are useful, please consider citing the paper.

```
@misc{wang2024neca3dcoronaryartery,
      title={NeCA: 3D Coronary Artery Tree Reconstruction from Two 2D Projections by Neural Implicit Representation}, 
      author={Yiying Wang and Abhirup Banerjee and Vicente Grau},
      year={2024},
      eprint={2409.04596},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2409.04596}, 
}
```

# 2. Introduction

Cardiovascular diseases (CVDs) are the most common health threats worldwide. 2D x-ray invasive coronary angiography (ICA) remains as the most widely adopted imaging modality for CVDs diagnosis. However, in current clinical practice, it is often difficult for the cardiologists to interpret the 3D geometry of coronary vessels based on 2D planes. Moreover, due to the radiation limit, in general only two angiographic projections are acquired, providing limited information of the vessel geometry and necessitating 3D coronary tree reconstruction based only on two ICA projections. In this paper, we propose a self-supervised deep learning method called NeCA, which is based on implicit neural representation using the multiresolution hash encoder and differentiable cone-beam forward projector layer in order to achieve 3D coronary artery tree reconstruction from two projections. We validate our method using six different metrics on coronary computed tomography angiography data in terms of right coronary artery and left anterior descending respectively. The evaluation results demonstrate that our NeCA method, without 3D ground truth for supervision and large datasets for training, achieves promising performance in both vessel topology preservation and branch-connectivity maintaining compared to the supervised deep learning model.

## Our Proposed Model Architecture

<p align="center">
  <img src="https://github.com/WangStephen/NeCA/blob/main/img/model.svg">
</p>

# 3. Packages Requirement

This work requires following dependency packages:

```
python: 3.8.17
pytorch: 1.9.0 
numpy: 1.24.4 
ninja: 1.11.1
PyYAML: 6.0
odl: 1.0.0.dev0
astra-toolbox: 2.1.0
tqdm: 4.65.0
argparse
```

# 4. Code Instructions

## Data Preparation

Please prepare your projection data in shape `(Batch, Number of projections, Height, Weight)` and put the data under the `./data/CCTA_test/` folder. We recommend to use [ODL](https://github.com/odlgroup/odl) to generate your simulated projections from 3D coronary tree, which is tested and incorporated in this work.

Then please update the corresponding projection geometry in the file `./data/config.yml`.

## Model Optimisation

Configure your model hyper-parameters in the file `./config/CCTA.yaml`.

Then run the model to start 3D reconstruction optimisation:

```
python train.py --config ./config/CCTA.yaml
```

The 3D reconstruction results during iterations are saved under the folder `./logs/`.

# 5. License

Please see [license](https://github.com/WangStephen/NeCA/blob/main/LICENSE).

# 6. Acknowledgement

NeCA model architectures are revised based on [NAF](https://github.com/Ruyi-Zha/naf_cbct) and [NeRP](https://github.com/liyues/NeRP).

Multi-resolution hash encoder is based on [torch-ngp](https://github.com/ashawkey/torch-ngp).

Differentiable cone-beam forward projection layer is based on [ODL](https://github.com/odlgroup/odl) and we also use it to generate our simulated projections to model for optimisation.
