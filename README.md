## MEgo2Vec: Embedding Matched Ego Networks for User Alignment Across Social Networks∗

This is basic implementation of our CIKM'18 paper:

Jing Zhang, Bo Chen, Xianming Wang, Hong Chen*, Cuiping Li, Fengmei Jin, Guojie Song, and Yutao Zhang. 2018. MEgo2Vec: Embedding Matched Ego Networks for User Alignment Across Social Networks. In Proceedings of ACM conference (CIKM’18).

## Requirements
- Ubuntu 16.04
- Python 2.7
- Tensorflow-gpu
- GPU,CUDA,CUDNN

Note: Running this project will consume upwards of 20GB hard disk space. The overall pipeline will take several hours. You are recommended to run this project on a Linux server.

## Data Description
Training data in this demo is about AMiner - Linkedin networks which is placed in the _data_ directory. If you want to download the original network data (AMiner, Linkedin), please use the link : https://pan.baidu.com/s/1b6_8jd8J9CGiCpyFBfZgoQ  密码:xacn .
If you want to get other networks (Twitter, MySpace LastFm...), please click the [link](https://www.aminer.cn/cosnet)

## How to run
```bash
cd code
python main.py
```
Note: Hyper parameter and training data in this demo is a little different than what we used in the experiments, so the performance (F1-score) will be a little bit lower than reported scores.
