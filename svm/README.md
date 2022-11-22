## Support Vector Machines

To Run the experiments `cd` into the `svm` directory and run:

```
$ ./run.sh
```

To set the parameters run
```
$ python3 experiments.py --lr={learning rate} --lr-a={a param for schedule} --C={penalty} --lr-schedule={lr schedule 'a' or 'b'}
```

To run the Dual run:
```
$ python3 experiments.py --dual --C={penalty}
```
