This repo contains the code and results of the AAAI 2021 paper:

<i><b> [Split then Refine: Stacked Attention-guided ResUNets for Blind Single Image Visible Watermark Removal](https://arxiv.org/abs/2012.07007)</b></i><br>
[Xiaodong Cun](http://vinthony.github.io), [Chi-Man Pun<sup>*</sup>](http://www.cis.umac.mo/~cmpun/) <br>
[University of Macau](http://um.edu.mo/)

[Datasets](#Resources) | [Models](#Resources) | [Paper](https://arxiv.org/abs/2012.07007)  | [🔥Online Demo!](https://colab.research.google.com/drive/1pYY7byBjM-7aFIWk8HcF9nK_s6pqGwww?usp=sharing)(Google CoLab)

<hr>

<img width="726" alt="nn" src="https://user-images.githubusercontent.com/4397546/101241905-37915d80-3735-11eb-9fb9-2e1e46d63f15.png">

<i>The overview of the proposed two-stage framework. Firstly, we propose a multi-task network, SplitNet, for watermark detection, removal,  and recovery. Then, we propose the RefineNet to smooth the learned region with the predicted mask and the recovered background from the previous stage. As a consequence, our network can be trained in an end-to-end fashion without any manual intervention. Note that, for clarity, we do not show any skip-connections between all the encoders and decoders.</i>
<hr>

> The whole project will be released in the January of 2021 (almost).


### Datasets

We synthesized four different datasets for training and testing, you can download the dataset via [huggingface](https://huggingface.co/datasets/vinthony/watermark-removal-logo/tree/main).

![image](https://user-images.githubusercontent.com/4397546/104273158-74413900-54d9-11eb-95fa-c6bee94de0ea.png)


### Pre-trained Models

* [27kpng_model_best.pth.tar (google drive)](https://drive.google.com/file/d/1KpSJ6385CHN6WlAINqB3CYrJdleQTJBc/view?usp=sharing)

> Other Pre-trained Models are still reorganizing and uploading, it will be released soon.


### Demos

An easy-to-use online demo can be founded in [google colab](https://colab.research.google.com/drive/1pYY7byBjM-7aFIWk8HcF9nK_s6pqGwww?usp=sharing).

The local demo will be released soon.

### Pre-requirements

```
pip install -r requirements.txt
```

### Train

Besides training our methods, here, we also give an example of how to train the [s2am](https://github.com/vinthony/s2am) under our framework. More details can be found in the shell scripts.


```
bash examples/evaluation.sh
```

### Test

```
bash examples/test.sh
```

## **Acknowledgements**
The author would like to thanks Nan Chen for her helpful discussion.

Part of the code is based upon our previous work on image harmonization [s2am](https://github.com/vinthony/s2am) 

## **Citation**

If you find our work useful in your research, please consider citing:

```
@misc{cun2020split,
      title={Split then Refine: Stacked Attention-guided ResUNets for Blind Single Image Visible Watermark Removal}, 
      author={Xiaodong Cun and Chi-Man Pun},
      year={2020},
      eprint={2012.07007},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## **Contact**
Please contact me if there is any question (Xiaodong Cun yb87432@um.edu.mo)
