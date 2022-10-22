#!/bin/sh
python3 optimize.py --mode='bgd'
python3 optimize.py --mode='sgd'
python3 optimize.py --mode='analytic'
