
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
To test the trained model on the CIFAR10 test set:
```sh
python3 test.py
```
### Qualitative evaluation
(top to bottom: original RGB, grayscale input, prediction)
#### Some good-looking samples:
![Screenshot](imgs/img1.png)

It is important to note that as a result of advarserial loss function, the goal of the generator is to fill the images with realistic/believable in order to fool the discriminator, rather than approximating the original colors of the image (altought, these can be equivalent sometimes). In this regard, the model performs quite well, often creating colorful and lifelike images.

#### Some Bad-looking ones:
![Screenshot](imgs/img2.png)

Evidently, the generator sometimes fails to identify a region or an object on an image, and it assigns unusual colors (at least to the our eyes) to these parts. It also has a tendency to sometimes create sepia-like or grayish images.


### Acknowledgements
 1. [The aforementioned paper](https://arxiv.org/abs/1803.05400), and the corresponding [github repo](https://github.com/ImagingLab/Colorizing-with-GANs).
 2. The excellent [PyTorch-GAN repository](https://github.com/eriklindernoren/PyTorch-GAN).

