# SeqViews2SeqLabels
This is the source code for our TIP paper **"SeqViews2SeqLabels: Learning 3D Global Features via Aggregating Sequential Views by RNN with Attention"**
![http://cgcad.thss.tsinghua.edu.cn/liuyushen/main/small/SeqViews2SeqLabels.png]()

## Requirements
+ python 2.7
+ tensorflow 1.0.12
+ numpy 1.13.3
+ scipy 0.19.0
+ matplotlib 2.0.1

Our code borrowed some code from `tensorflow.contrib.legacy_seq2seq`, but the newer version of tensorflow has moved this package to `tensorflow.contrib.seq2seq`, so to run this code, please make sure 
your version of tensorflow has this package(we recomend to use 1.0.12 as ours).

## Datasets
+ [ModelNet](http://modelnet.cs.princeton.edu/): ModelNet10, ModelNet40
+ [ShapeNet55](https://www.shapenet.org/)


## Codes
+ `run.py` execute train or test command.
+ `train.py` train and test SeqViews2SeqLabels model.
+ `seq_rnn_model.py` the SeqViews2SeqLabels model.
+ `model_data.py` read data as required format of model inputs.
+ `utils` utils for visualization and retrieval.

## Usage
To run this code, please go with below steps:
1. Extract feature for views of each spilt of dataset with VGG19, the features are numpy matrix with shape  [n_shapes, n_views=12, 4096], in total we need:
	+ modelnet10 train feature, test feature
	+ modelnet40 train feature, test feature
	+ shapenet55 train feature, val feature, test feature 
2. Prepare labels as one dimensional numpy array [n_shapes], the order of labels should be corrsponded with the features, in total we need:
	+ modelnet10 train labels, test labels
	+ modelnet40 train labels, test labels
	+ shapenet55 train labels, val labels, test labels
3. Modify the `data_patha` in `run.py`:
	the paths for each dataset are gived as below:
	train feature file, train labels file, test feature file, test labels file, result dir for saving the trained model and log, result file for saving test result

the run command is `python run.py --dataset=<dataset> --train=<train>`, 
	the options for `dataset` parameter are **modelnet10**, **modelnet40**, **shapenet55**,
	and for `train` are **True** or **False**

For example, to train on modelnet10, run the command `python run.py --dataset=modelnet10 --train=True`;
to test on shapenet55, run the command `python run.py --dataset=shapenet55 --train=False`.

## Citation
If you find this useful, please cite our work as follows:
```
@article{han2018SeqViews2SeqLabels,
  title={SeqViews2SeqLabels: Learning 3D Global Features via Aggregating Sequential Views by RNN with Attention},
  author={Zhizhong Han, Mingyang Shang, Zhenbao Liu, Chi-Man Vong, Yu-Shen Liu, Junwei Han, Matthias Zwicker, C.L. Philip Chen},
  journal={IEEE Transactions on Image Processing},
  year={2018},
  publisher={IEEE}
}
```

