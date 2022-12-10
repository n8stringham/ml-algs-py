#!/bin/sh

# Experiments to vary the activation, initialization, width, and num_layers.
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=5 --num_layers=3
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=5 --num_layers=5
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=5 --num_layers=9
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=10 --num_layers=3
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=10 --num_layers=5
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=10 --num_layers=9
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=25 --num_layers=3
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=25 --num_layers=5
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=25 --num_layers=9
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=50 --num_layers=3
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=50 --num_layers=5
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=50 --num_layers=9
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=100 --num_layers=3
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=100 --num_layers=5
python3 experiments.py --mode=torch --activation=ReLU --initialization=He --width=100 --num_layers=9

# tanh + Xavier
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=5 --num_layers=3
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=5 --num_layers=5
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=5 --num_layers=9
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=10 --num_layers=3
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=10 --num_layers=5
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=10 --num_layers=9
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=25 --num_layers=3
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=25 --num_layers=5
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=25 --num_layers=9
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=50 --num_layers=3
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=50 --num_layers=5
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=50 --num_layers=9
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=100 --num_layers=3
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=100 --num_layers=5
python3 experiments.py --mode=torch --activation=tanh --initialization=Xavier --width=100 --num_layers=9
