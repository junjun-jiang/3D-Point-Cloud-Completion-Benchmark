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

- **[Shape-Inversion]** Unsupervised 3D Shape Completion through GAN Inversion, CVPR2021,J. Zhang et al. [[PDF]](https://arxiv.org/abs/2104.13366)[[Code]](https://github.com/junzhezhang/shape-inversion)
- **[ASFM-Net]** ASFM-Net: Asymmetrical Siamese Feature Matching Network for Point Completion,  ACM MM2021, Y. Xia et al. [[PDF]](https://arxiv.org/abs/2104.09587)[[Code]]( https://github.com/Yan-Xia/ASFM-Net)
- **[PoinTr]** PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers, ICCV 2021, X. Yu et al. [[PDF]](https://arxiv.org/abs/2108.08839)[[Code]](https://github.com/yuxumin/PoinTr)

### 2020

- **[PF-Net]** PF-Net: Point Fractal Network for 3D Point Cloud Completion, CVPR2020, Z. Huang et al. [[PDF]](https://arxiv.org/abs/2003.00410v1)[[Code]](https://github.com/zztianzz/PF-Net-Point-Fractal-Network)
- **[CRN]** Cascaded Refinement Network for Point Cloud Completion with Self-supervision, CVPR2020, X. Wang et al. [[PDF]](https://arxiv.org/abs/2010.08719v1)[[Code]](https://github.com/xiaogangw/cascaded-point-completion)
- **[MSN]** Morphing and Sampling Network for Dense Point Cloud Completion, AAAI2020, M. Liu et al. [[PDF]](https://arxiv.org/abs/1912.00280)[[Code]](https://github.com/TheoDEPRELLE/MSN-Point-Cloud-Completion)
- **[SFA]** Detail Preserved Point Cloud Completion via Separated Feature Aggregation, ECCV 2020, W. Zhang et al. [[PDF]](https://arxiv.org/abs/2007.02374)[[Code]](https://github.com/XLechter/Detail-Preserved-Point-Cloud-Completion-via-SFA)
- **[GRNet]** 3D Point Capsule Networks, ECCV2020, H. Xie et al. [[PDF]](https://arxiv.org/abs/2006.03761v1)[[Code]](https://github.com/hzxie/GRNet)


### 2019

- **[3D Capsule]** 3D Point Capsule Networks, CVPR2019, Y. Zhao et al. [[PDF]](https://arxiv.org/pdf/1812.10775)[[Code]](https://tinyurl.com/yxq2tmv3)
- **[TopNet]** TopNet: Structural Point Cloud Decoder, CVPR2019, Lyne P. Tchapmi et al. [[PDF]](https://ieeexplore.ieee.org/document/8953650)[[Code]](http://completion3d.stanford.edu)

### 2018
- **[PCN]** PCN: Point completion network, 3DV2018, W. Yuan et al. [[PDF]](https://arxiv.org/abs/1808.00671)[[Code]](https://github.com/wentaoyuan/pcn)
- **[l-GAN]** Learning Representations and Generative Models for 3D Point Clouds, ICML2018,P. Achlioptas et al. [[PDF]](https://arxiv.org/abs/1707.02392)[[Code]](http://github.com/optas/latent_3d_points)

### 2017
- **[CNN Complete]** Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis, CVPR2017, A. Dai et al. [[PDF]](https://arxiv.org/abs/1612.00101v2)[[Code]](https://github.com/star-cold/cnncomplete)

### Before 2017

-








