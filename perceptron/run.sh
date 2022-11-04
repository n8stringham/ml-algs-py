#!/bin/sh

#echo Running standard perceptron with 3 learning rates
# std perceptron with 3 learning rates
#python3 experiments.py --model=std --lr=1 --epochs=10
#python3 experiments.py --model=std --lr=.5 --epochs=10
python3 experiments.py --model=std --lr=.1 --epochs=10

#echo Running voted perceptron with 3 learning rates
# voted perceptron with 3 learning rates
#python3 experiments.py --model=voted --lr=1 --epochs=10
#python3 experiments.py --model=voted --lr=.5 --epochs=10
python3 experiments.py --model=voted --lr=.1 --epochs=10

#echo Running avgeraged perceptron with 3 learning rates
# averaged perceptron with 3 learning rates
#python3 experiments.py --model=avg --lr=1 --epochs=10
#python3 experiments.py --model=avg --lr=.5 --epochs=10
python3 experiments.py --model=avg --lr=.1 --epochs=10
