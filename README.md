# 3D-Point-Cloud-Completion-Benchmark
A list of 3D point cloud completion resources. We try to keep it updated every week or two with the latest papers.


## Datasets
#### Synthetic datasets
- PCN dataset
- ShapeNet Benchmark
- Completion3D benchmark dataset

#### Real-world datasets
- KITTI dataset

## Evaluation Metrics
- Chamfer Distance (CD) 
- Fréchet Point Cloud Distance (FPD)
- Earth Mover Distance (EMD)
- Accuracy: Accuracy measures the fraction of points in the output that are matched with the ground truth
- Completeness: Similar to accuracy, completeness reports the fraction of points in the ground truth that are within a distance threshold to any point in the output.
- F-score: F-score is calculated as the harmonic average of the accuracy and completeness.
- Fidelity. Fidelity measures how well the inputs are preserved in the outputs.
- Plausibility. Plausibility is evaluated as the classification accuracy in percentage by a pre-trained PointNet model.

## Papers

### 2021

-

### 2020

- **[PF-Net]** PF-Net: Point Fractal Network for 3D Point Cloud Completion, CVPR2020, Z. Huang et al. [[PDF]](https://arxiv.org/abs/2003.00410v1)[[Code]](https://github.com/zztianzz/PF-Net-Point-Fractal-Network)
- **[CRN]** Cascaded Refinement Network for Point Cloud Completion with Self-supervision, CVPR2020, X. Wang et al. [[PDF]](https://arxiv.org/abs/2010.08719v1)[[Code]](https://github.com/xiaogangw/cascaded-point-completion)
- **[MSN]** Morphing and Sampling Network for Dense Point Cloud Completion, AAAI2020, M. Liu et al. [[PDF]](https://arxiv.org/abs/1912.00280)[[Code]](https://github.com/TheoDEPRELLE/MSN-Point-Cloud-Completion)


### 2019

- **[3D Capsule]** 3D Point Capsule Networks, CVPR2019, Y. Zhao et al. [[PDF]](https://arxiv.org/pdf/1812.10775)[[Code]](https://tinyurl.com/yxq2tmv3)

### 2018
- **[PCN]** PCN: Point completion network, 3DV2018, W. Yuan et al. [[PDF]](https://arxiv.org/abs/1808.00671)[[Code]](https://github.com/wentaoyuan/pcn)
- **[l-GAN]** Learning Representations and Generative Models for 3D Point Clouds,ICML2018,P. Achlioptas et al. [[PDF]](https://arxiv.org/abs/1707.02392)[[Code]](http://github.com/optas/latent_3d_points)

### 2017

-

### Before 2017

-








