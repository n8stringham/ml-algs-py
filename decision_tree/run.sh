#!/bin/bash

python3 experiments.py --train=../data/car-data/train.csv --test=../data/car-data/test.csv
python3 experiments.py --train=../data/bank-data/train.csv --test=../data/bank-data/test.csv

