# Tensorflow 2.0 GAN library

## Introduction
Implementation of GAN papers, all using cifar10 dataset in this project.

ATTENTION:
To compare the differences of GAN methods, the hyperparameters in this project are not exactly same as papers.
Architecture of generators and discriminators are as similar as possible, and using same optimizer setting.

## Environment

```
python==3.6
tensorflow==2.0
```

##  Implemented Papers

 - DCGAN - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks [link](https://arxiv.org/abs/1511.06434)
 - LSGAN - Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities [link](https://arxiv.org/abs/1701.06264)
 - WGAN-GP - Improved Training of Wasserstein GANs [link](https://arxiv.org/abs/1704.00028)
 - SNGAN - Spectral Normalization for Generative Adversarial Networks [link](https://arxiv.org/abs/1802.05957)
 - SAGAN - Self-Attention Generative Adversarial Networks [link](https://arxiv.org/abs/1805.08318)
 
## Results

| Name | 50 epochs |
| :---: | :---: |
| DCGAN | ![alt text](https://i.imgur.com/DupOZSn.png "DCGAN") |
| LSGAN | ![alt text](https://i.imgur.com/fEn0eFO.png "LSGAN") |
| WGAN-GP | ![alt text](https://i.imgur.com/n6egM1D.png "WGAN-GP") |
| SNGAN | ![alt text](https://i.imgur.com/sMXuAeD.png "SNGAN") |
| SAGAN | ![alt text](https://i.imgur.com/akHOQBd.png "SAGAN") |