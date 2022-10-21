# description of how to run the linear regression code.
# Linear Regression
This code implements 3 common optimization routines for finding the weights in linear regression. 

## Running the Code
To run the experiments for CS6350, make sure you are in the `linear_regression` directory and then run
```
$ ./run.sh
```
This will print the results for each experiment to stdout.




If you are interested in just running a specific optimization routine then use the command.

```
$ python3 optimize.py --mode={optimization_routine}
```
where optimization\_routine is one of {'bgd', 'sgd', 'analytic'}

Note: this will run the optimization for the [SLUMP](https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test) dataset.

Brief descriptions of each of these modes are are found below.

## Batch Gradient Descent
This is the traditional gradient based optimization method in which the weight vector is updated after computing the gradient vector for the entire training set. These updates continue until convergence.

## Stochastic Gradient Descent
This is a common method which speeds up the convergence of gradient descent by first randomly shuffling the training examples. The algorithm then uses one training example at a time to compute the gradient and update the weights. This proceeds for a specified number of iterations until convergence.

# Analytic Closed Form Solution
There is also a closed form solution for this problem for finding the optimal weights. This is implemented using matrices.

