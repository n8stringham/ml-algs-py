## Perceptron
This program implements 3 variants of the Perceptron (Standard, Voted, and Averaged)

To run the experiments for HW3 please cd into this directory and then run the script
```
$ ./run.sh
```

To train initialize a Perceptron model with different parameters run
```
python3 experiments.py --model={std, voted, avg} --lr={0 < float < 1} --epochs={int}


