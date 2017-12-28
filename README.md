# Chainer implementation of Commonsense Knowledge Base Completion

Chainer-based implementation of Commonsense Knowledge Base Completion.

See "[Commonsense Knowledge Base Completion](http://aclweb.org/anthology/P/P16/P16-1137.pdf)", Xiang Li; Aynaz Taheri; Lifu Tu; Kevin Gimpel, ACL 2016.

# Development Environment

* Ubuntu 16.04
* Python 3.5.2
* Chainer 3.1.0
* numpy 1.13.3
* cupy 2.1.0
* nltk
* progressbar
* and their dependencies

# Dataset

Dataset is derived from [the original implementation](https://github.com/Lorraine333/ACL_CKBC).
Now you have seven files:
* new_omcs100.txt: 100k facts which are used for training.
* new_omcs300.txt: 300k facts which are used for training.
* new_omcs600.txt: 600k facts which are used for training.
* new_omcs_dev1.txt: 1.2k facts which are used for learning a threshold to classify true facts and false facts.
* new_omcs_dev2.txt: 1.2k facts which are used for measuring generalization ability during training.
* new_omcs_test.txt: 2.4k facts which are used for measuring generalization ability.
* rel.txt: A list of vocabulary of relations.

# How to Run

First, you need to prepare vocabulary of concepts.

```
$ python preprocess.py <path/to/training-data> --n-vocab <number of vocabulary>
```

Then, let's start training.

```
$ python train.py <path/to/training-data> <path/to/concept-vocabulary> <path/to/relation-vocabulary> --validation <path/to/validation-data> --embedding <path/to/pretrained-word-embedding> -g <gpu-id>
```

See command line help for other options.
