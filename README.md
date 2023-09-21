# human-in-the-loop framework
The Pytorch implementation of the paper: An Interactive Framework of Balancing Evaluation Cost and Prediction Accuracy for Knowledge Tracing.

## Setup
We use an open-source platform [PYKT](https://www.pykt.org) as a basis.

The specific installation is as follows:
```
cd pykt-toolkit
pip install -e .
```

## Data Preprocessing
Downloading the dataset and placing it in the `data/{dataset_name}` folder.
```
cd examples
python data_preprocess.py --dataset_name=ednet
```

## Training for the proposed model
Here is an example of the training:
```
cd examples
CUDA_VISIBLE_DEVICES=0 nohup python -u wandb_rakt_train.py --dataset_name=nips_task34 --embed_size=64 --num_attn_layers=2 --num_heads=4 --drop_prob=0.1 --fold=2 --learning_rate=0.001 --seed=3407 --use_wandb=0 --add_uuid=0 > rakt_check.txt &
```
