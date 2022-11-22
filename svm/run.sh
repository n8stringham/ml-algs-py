#!/bin/sh
echo ------------
echo Experiment 2a
echo ------------
# C = 100/873 => .1145
python3 experiments.py --lr=.01 --lr-a=.01 --C=.1145 --lr-schedule='a'
echo
# C = 500/873 => .5272
python3 experiments.py --lr=.01 --lr-a=.01 --C=.5272 --lr-schedule='a'
echo
# C = 700/873 => .8018
python3 experiments.py --lr=.01 --lr-a=.01 --C=.8018 --lr-schedule='a'
echo
echo ------------
echo Experiment 2b
echo ------------
# C = 100/873 => .1145
python3 experiments.py --lr=.01 --lr-a=.01 --C=.1145 --lr-schedule='b'
echo
# C = 500/873 => .5272
python3 experiments.py --lr=.01 --lr-a=.01 --C=.5272 --lr-schedule='b'
echo
# C = 700/873 => .8018
python3 experiments.py --lr=.01 --lr-a=.01 --C=.8018 --lr-schedule='b'
echo ------------
echo Experiment 2c
echo ------------
python3 experiments.py --lr=.01 --lr-a=.01 --C=.1145 --diff
echo
python3 experiments.py --lr=.01 --lr-a=.01 --C=.5272 --diff
echo
python3 experiments.py --lr=.01 --lr-a=.01 --C=.8018 --diff
echo

echo ------------
echo Experiment 3a
echo ------------
python3 experiments.py --dual --C=.1145 --kernel=linear
echo
python3 experiments.py --dual --C=.5272 --kernel=linear
echo
python3 experiments.py --dual --C=.8018 --kernel=linear

echo ------------
echo Experiment 3b
echo ------------
python3 experiments.py --dual --C=.1145 --kernel=gaussian --gamma=.1
python3 experiments.py --dual --C=.1145 --kernel=gaussian --gamma=.5
python3 experiments.py --dual --C=.1145 --kernel=gaussian --gamma=1
python3 experiments.py --dual --C=.1145 --kernel=gaussian --gamma=5
python3 experiments.py --dual --C=.1145 --kernel=gaussian --gamma=100
echo
python3 experiments.py --dual --C=.5272 --kernel=gaussian --gamma=.1
python3 experiments.py --dual --C=.5272 --kernel=gaussian --gamma=.5
python3 experiments.py --dual --C=.5272 --kernel=gaussian --gamma=1
python3 experiments.py --dual --C=.5272 --kernel=gaussian --gamma=5
python3 experiments.py --dual --C=.5272 --kernel=gaussian --gamma=100
echo
python3 experiments.py --dual --C=.8018 --kernel=gaussian --gamma=.1
python3 experiments.py --dual --C=.8018 --kernel=gaussian --gamma=.5
python3 experiments.py --dual --C=.8018 --kernel=gaussian --gamma=1
python3 experiments.py --dual --C=.8018 --kernel=gaussian --gamma=5
python3 experiments.py --dual --C=.8018 --kernel=gaussian --gamma=100
echo


echo ------------
echo Experiment 3c
echo ------------
python3 experiments.py --dual --C=.1145 --kernel=gaussian --gamma=.1 --overlap
python3 experiments.py --dual --C=.5727 --kernel=gaussian --gamma=.1 --overlap
python3 experiments.py --dual --C=.8018 --kernel=gaussian --gamma=.1 --overlap

