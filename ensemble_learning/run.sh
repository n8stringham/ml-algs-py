#!/bin/bash
python3 adaboost.py --train=../data/bank-data/train.csv --test=../data/bank-data/test.csv --T=500

python3 bagging.py --train=../data/bank-data/train.csv --test=../data/bank-data/test.csv --T=500

python3 rf.py --train=../data/bank-data/train.csv --test=../data/bank-data/test.csv --T=500
