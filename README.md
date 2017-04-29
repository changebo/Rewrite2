# Notes:

```
python preprocess.py --source_font fonts/hanyiluobo.ttf --target_font fonts/huawenxinwei.ttf --char_list charsets/top_3000_simplified.txt --save_dir fonts
python preprocess.py --source_font fonts/simhei.ttf --target_font fonts/hanyiheiqi.ttf --char_list charsets/top_3000_simplified.txt --save_dir fonts
python preprocess.py --source_font fonts/SIMSUN.ttf --target_font fonts/simkai.ttf --char_list charsets/top_3000_simplified.txt --save_dir fonts --char_size 64 --canvas 64

rm -rf checkpoint*

python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/huawenxinwei.npy --sample_dir=samples_huawenxinwei --checkpoint_dir=checkpoint_huawenxinwei --epoch=100 --is_train
python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/hanyiluobo.npy --sample_dir=samples_hanyiluobo --checkpoint_dir=checkpoint_hanyiluobo --epoch=100 --is_train
python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/hanyiheiqi.npy --sample_dir=samples_hanyiheiqi --checkpoint_dir=checkpoint_hanyiheiqi --epoch=100 --is_train
python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/simhei.npy --sample_dir=samples_simhei --checkpoint_dir=checkpoint_simhei --epoch=100 --is_train

python main.py --source_font=fonts/SIMSUN.npy --target_font=fonts/simkai.npy --sample_dir=samples_simkai --checkpoint_dir=checkpoint_simkai --epoch=100 --tv_penalty=0 --L1_penalty=0 --is_train --source_height=64 --source_width=64 --target_height=64 --target_width=64


```

kai -> hei

[source, G(source)]
[source, source]
[source, target]


# DCGAN in Tensorflow

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

![alt tag](DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper.*


## Online Demo

[<img src="https://raw.githubusercontent.com/carpedm20/blog/master/content/images/face.png">](http://carpedm20.github.io/faces/)

[link](http://carpedm20.github.io/faces/)


## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) [moviepy](https://github.com/Zulko/moviepy) (for visualization)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset


## Usage

First, download dataset with:

    $ python download.py --datasets mnist celebA

To train a model with downloaded dataset:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --c_dim=1 --is_train
    $ python main.py --dataset celebA --input_height=108 --is_train --is_crop True

To test with an existing model:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --c_dim=1
    $ python main.py --dataset celebA --input_height=108 --is_crop True

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --is_train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --c_dim=1 --is_train

## Results

![result](assets/training.gif)

### celebA

After 6th epoch:

![result3](assets/result_16_01_04_.png)

After 10th epoch:

![result4](assets/test_2016-01-27%2015:08:54.png)

### Asian face dataset

![custom_result1](web/img/change5.png)

![custom_result1](web/img/change2.png)

![custom_result2](web/img/change4.png)

### MNIST

MNIST codes are written by [@PhoenixDai](https://github.com/PhoenixDai).

![mnist_result1](assets/mnist1.png)

![mnist_result2](assets/mnist2.png)

![mnist_result3](assets/mnist3.png)

More results can be found [here](./assets/) and [here](./web/img/).


## Training details

Details of the loss of Discriminator and Generator (with custom dataset not celebA).

![d_loss](assets/d_loss.png)

![g_loss](assets/g_loss.png)

Details of the histogram of true and fake result of discriminator (with custom dataset not celebA).

![d_hist](assets/d_hist.png)

![d__hist](assets/d__hist.png)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
