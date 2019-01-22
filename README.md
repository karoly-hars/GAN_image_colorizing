
# Image colorization with GANs

The aim of this project is to explore the topic of image colorization with the help of Generative Adversarial Networks.


### Approach
The project heavly builds on the findings of the paper [Image Colorization with Generative Adversarial Networks by Nazeri et al.](https://arxiv.org/abs/1803.05400). Similarly to the article, I studied image colorization on the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) with the help of adversarial learning. The generator network is based on a UNet, and the discriminator matches the generator's contractive path.

### Prerequisites
 - numpy
 - torch 1.0.0
 - opencv-python
 - matplotlib

### Running the code
To run the training code:
```sh
python3 train.py
```
To download the trained generator weights, for inferencing on the CIFAR10 test set:
```sh
python3 test.py
```
### Qualitative evaluation
(top to bottom: original RGB, grayscale input, prediction)
#### Some good-looking samples:
![](imgs/img1.png =1174x256)

#### Some Bad-looking ones:
![](imgs/img2.png =567x256)


### Acknowledgements
 1. [The aforementioned paper](https://arxiv.org/abs/1803.05400), and the corresponding [github repo](https://github.com/ImagingLab/Colorizing-with-GANs).
 2. The excellent [PyTorch-GAN repository](https://github.com/eriklindernoren/PyTorch-GAN).

