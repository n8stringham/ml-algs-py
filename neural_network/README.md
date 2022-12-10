## Feed Forward Nerual Network

To run the experiments for homework `cd` into the `nerual_network` directory and run the appropriate commands below.


## NN from scratch in python
```
$ ./run_network.sh
```

## NN using Pytorch
```
$ ./run_network_bonus.sh
```


# Setting params individually
You can run my python implementation of the NN with parameters of your choosing like this:

```
$ python3 network.py --epochs={int} --gamma={float} --d={float} --init={'random' or 'zero'} --hidden_dim={int}
```

This will train a Feed Forward Neural Net with 2 hidden layers of your specified size on the bank-note dataset from the UCI Machine Learning Repo.
