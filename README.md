# Bilingual Document Classification

Train a text classifier with French and English documents, by using bilingual word embedding. There are three steps:

## 1 Training a bilingual word embedding
* https://github.com/facebookresearch/MUSE.git
* https://github.com/Babylonpartners/fastText_multilingual
* http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html

## 2 Data preprocessing
Using the pre-train bilingual word embedding to embed the bilingual documents
```
$ python preprocess.py

The following arguments are mandatory:
  --source_path              Path to raw data, filename should be in the format of "en_train.tsv"
  --embedding_path           Path to pre-trained bilingual word embedding

The following arguments are optional:
  --train                    Filename of the preprocessed train set ['train_muse.dat']
  --en_test                  Filename of the preprocessed English test set ['en_test_muse.dat']
  --fr_test                  Filename of the preprocessed French test set ['fr_test_muse.dat']
  --preprocessed_path        Path to preprocessed data ['../preprocessed_data/']
```

## 3 Train
```
$ python train.

The following arguments are optional:
  --load_net                 Path to trained model (For continue training) ['']
  --n_hidden                 Feature extraction layer dimension [2048]
  --n_mlp_layers             Number of MLP layers [1]
  --batch_size               Batch size [124]
  --n_epoch                  Number of maximum training epochs [1000]
  --optimizer                Optimizer type: must choose RMSprop, SGD or Adam. [RMSprop]
  --lr                       Learning rate [0.0001]
  --dropout                  Probability of dropout [0.8]
  --lr_anneal                Parameter for learning rate annealing [1], or can choose to use lr_scheduler by default
```




