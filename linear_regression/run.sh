#!/bin/sh

python optimize.py --mode='bgd'
python optimize.py --mode='sgd'
python optimize.py --mode='analytic'
