
You can run all experiments with

```
$ ./run.sh
```

It will take a VERY long time to run all of the experiments!
Here are the individual commands to run for specific portions of the assignment

## AdaBoost - problem 2A
```
$ python3 adaboost.py --train=../data/bank-data/train.csv --test=../data/bank-data/test.csv --T=500
```
## Bagging - problem 2B, 2C
```
$ python3 bagging.py --train=../data/bank-data/train.csv --test=../data/bank-data/test.csv --T=500
```
## Random Forest - problem 2D, 2E
```
$ python3 rf.py --train=../data/bank-data/train.csv --test=../data/bank-data/test.csv --T=500
```
