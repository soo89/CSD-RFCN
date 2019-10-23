# CSD: Consistency-based Semi-supervised learning for object Detection

By [Jisoo Jeong](http://mipal.snu.ac.kr/index.php/Jisoo_Jeong), [Seungeui Lee](http://mipal.snu.ac.kr/index.php/Seungeui_Lee), [Jee-soo Kim](http://mipal.snu.ac.kr/index.php/Jee-soo_Kim), [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak)



## Installation & Preparation
We experimented with CSD using the RFCN pytorch framework. To use our model, complete the installation & preparation on the [RFCN pytorch homepage](https://github.com/princewang1994/R-FCN.pytorch)

## Check list
```Shell
check DATA_DIR in '/lib/model/utils/config.py'
```

## Supervised learning
```Shell
python train_rfcn.py
```

## CSD training
```Shell
python train_csd.py
```

## Evaluation
```Shell
python test_net.py
```
