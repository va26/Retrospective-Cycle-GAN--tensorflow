Retrospective-Cycle-GAN--tensorflow
===================================

Tensorflow implementation for the paper [Predicting Future Frames using Retrospective Cycle GAN, 2019][1].

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Download and Preprocess Data](#download-and-preprocess-data)
  * [Training](#training)
  * [Results](#results)
  * [Author](#author)
  * [Acknowledgements](#acknowledgements)


# Introduction

The project is an attempt to implement Generative Adversarial Network (GAN) mentioned in the [paper][1]. The central theme of the work by the authors is to incorporate the training of a single generator that can predict both future and past frames while enforcing the consistency of bi-directional prediction using the retrospective cycle constraints. The authors suggest to employ two discriminators not only to identify fake frames but also to distinguish fake contained image sequences from the real sequence. The authors claim that their method showed state-of-the-art performance in predicting the future frames. Due to limited resources, I trained my network on a subset of the original [UCF-101][8] dataset. From the results shown below, we can infer that the outcome is close to the results presented in the paper.


# Setup and Dependencies

This repository is only compatible with Python3 and following are the list of dependencies to make this repository work:

1. Tensorflow 2.2
2. CUDA 10.1
3. CuDNN 7.6
4. FFMPEG

**NOTE**: A detailed list of the dependencies are mentioned in the [requirements.txt][11] file.

See the [Getting Started](#getting-started) section on how to setup the environment. For CUDA and CuDNN installation refer to the [article][13] in **towardsdatscience** and [nvidia docs][14].

## Getting Started

This starter code is implemented using Tensorflow v2.2, and provides out of the box support with CUDA 10.1 and CuDNN 7.6.
There are two recommended ways to set up this codebase: Anaconda or Miniconda.

### Anaconda or Miniconda

1. Install Anaconda or Miniconda distribution based on Python3.5+ from their [downloads' site][2].
2. Clone this repository and create an environment:

```sh
git clone https://github.com/va26/Retrospective-Cycle-GAN--tensorflow.git
conda create -n retrogan python=3.7

# Activate the environment and install all dependencies
conda activate retrogan
cd Retrospective-Cycle-GAN--tensorflow/
pip install -r requirements.txt
```

3. Install ffmpeg from [website][3]
Setup the ffmpeg application in your system path (For window users, steps to follow: [link][4])

```
#install ffmpeg using pip in the venv created above
pip install ffmpeg
```


# Download and Preprocess Data

1. Download the UCF-101 video files from [here][8] and keep it under `$PROJECT_ROOT/UCF-101` directory, for default arguments to work effectively. I have uploaded an example dataset. Here `$PROJECT_ROOT` refers to root directory ofthe project.

2. Use the `image_extractor.py` file to extract video frames from source folder(s) to their respective destination folder(s). It will recursively traverse the folders and extract frames at given intervals and store them to their corresponding folders under the parent directory `./UCF-101/img_data/`.
```sh
python image_extractor.py
```

3. Once data extraction is complete, Pre-process the data into aggregated chunks as follows:
```sh
python prep_data.py --chunk_size
```
Here the `chunk_size` is used to create pickle dumps of aggregated batches. Size of each chunk, keep `2**x` based on the **dataset, no.of GPUs and their capacity.** 
```
Example: if chunk_size = 128 and image dimensions are 240x320x3 (height*width*(RGB channels)), then pkl dumps will be:
(240, 320, 3, 5, 128), where 5 is no. of images stacked together
```


# Training

This codebase supports training on multiple GPUs using the [distributed mirrored strategy][6] in tensorflow, try out specifying GPU ids to train scripts as: `--gpu-ids 0 1 2 3`

The training script provided allows the user to enter arguments for training such as:
1. batch_size: (integer) <=2 (in the paper they have used batch size as 1, hence default is 1)
2. max_ckpt: (integer) Maximum checkpoints to save while training (default is 5)
3. epoch: (integer) Total no. of epochs for training (default is 100)
4. filter_size (optional): (integer) Filter size for LoG filter (Laplacian of Gaussian). Refer [this][7] for more details on LoG.
6. sigma (optional): (integer) Sigma for LoG filter

Train the model provided in this repository as:

```sh
python train.py --batch 2 --max-ckpt 3 --epoch 30 --filter-size 4 --sigma 2 --gpu-ids 0 1 # provide more ids for multi-GPU execution
```

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--ckpt-dir`. Refer [va26/Retrospective-Cycle-GAN--tensorflow/blob/main/train.py][10] *(Line 111)* for more details on how checkpointing is managed.

### Logging

We use [Tensorboard][5] for logging training progress. Recommended: execute `tensorboard --logdir /path/to/tb-log-dir --port 8008` and visit `localhost:8008` in the browser. Note `--tb-log-dir` can be provided as argument by the user otherwise it saves to default dir i.e. `$PROJECT_ROOT/tensorboard`


# Results

The training was done for **500 epochs** on a *subset* of the complete dataset due to the lack of compute resources. Once both generator and discriminator losses were sufficiently low, the stored model checkpoints were used to predict the future frame. Results of the experiment as shown below:

![first_image](https://github.com/va26/Retrospective-Cycle-GAN--tensorflow/blob/main/Images/Model_output_wm_marked.png)

*As can been seen from the image above, the highlighted areas (in predicted frame and ground truth) show that training on the subset is not enough to output accurate results and there is scope for improvement if trained on complete dataset or the quality of images used for training are of higher resolution.*

The **predicted frame** is less distinguishable from the **ground truth** in images capturing low motion scenarios. The images do appear to be smoothened i.e. the sharpness is reduced, when compared closely with the ground truth, which leaves scope for future work to improve upon this.

![second_image](https://github.com/va26/Retrospective-Cycle-GAN--tensorflow/blob/main/Images/Model_output_1_wm_cropped.png)


# Acknowledgements

* Some segments of the code were inspired from the implementation given in [Pix2pix][12].


# Author

Vatsal Aggarwal


[1]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Kwon_Predicting_Future_Frames_Using_Retrospective_Cycle_GAN_CVPR_2019_paper.pdf
[2]: https://conda.io/docs/user-guide/install/download.html
[3]: https://www.ffmpeg.org/download.html
[4]: https://www.wikihow.com/Install-FFmpeg-on-Windows
[5]: https://www.github.com/lanpa/tensorboardX
[6]: https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy
[7]: https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian
[8]: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
[10]: https://github.com/va26/Retrospective-Cycle-GAN--tensorflow/blob/main/train.py
[11]: https://github.com/va26/Retrospective-Cycle-GAN--tensorflow/blob/main/requirements.txt
[12]: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
[13]: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
[14]: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
