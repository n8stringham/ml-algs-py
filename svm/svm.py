# This program implements the SVM class
import numpy as np
from scipy import optimize

class SVM():
    def __init__(self, epochs=100, lr=.01, C=100/873, schedule='a', a=1):
        self.epochs = epochs
        self.lr = lr
        self.schedule = schedule
        self.a = a
        self.C = C
        self.N = None
        self.w = None

    def _schedule(self, lr, a, t, schedule):
        '''
        Return lr to be used for current epoch given a schedule.
        '''
        if schedule == 'a':
            return lr / (1 + (lr / a)*t)

        elif schedule == 'b':
            return lr / (1 + t)

    def _shuffle(self, X, y):
        '''
        randomly shuffle in unison the train instances and the labels.
        '''
        assert len(X) == len(y)
        rng = np.random.default_rng()
        perm = rng.permutation(len(X))
        shuffled_X = X[perm] 
        shuffled_y = y[perm]
        return X, y

    def train(self, X, y):
        '''
        train the svm model using stochastic subgradient descent.
        '''
        objectives = []
        steps = []

        # fold in bias term
        X = self._fold_bias(X)
        self.N = X.shape[0]

        # params
        w_init = np.zeros(X.shape[1])
        w = w_init

        # train loop
        for t in range(self.epochs):
            # randomly shuffle the data
            X, y = self._shuffle(X,y)
            
            # set the learning rate for this epoch
            lr = self._schedule(self.lr, self.a, t, self.schedule)
            for instance, label in zip(X, y):
                grad = self._subgrad(instance, label, w)
                #print("grad=",grad)
                w -= lr * grad
                #print("w=",w)
            # compute the loss for this epoch
            objective = self._objective(w, X, y)
            objectives.append(objective)
            steps.append((t+1)*len(X))
            #print("objective=",objective)

        self.w = w
        return objectives, steps

    def _objective(self, w, X, y):
        '''
        compute the value of the objective function for svm.
        '''
        #print("w.shape=",w.shape)
        #print("w=",w)
        #print("X.shape=",X.shape)
        #print("y.shape=",y.shape)
        hinge_loss = max(0, np.sum(1 - y * (X @ w)))
        #print("hinge_loss=",hinge_loss)
        return (1/2) * w.T @ w + self.C * hinge_loss


    def _fold_bias(self, X):
        '''
        Add new element with value of 1 to end of each training instance to represent the bias.
        '''

        b = np.ones((X.shape[0], 1))

        # fold in bias term: don't use b anymore
        X = np.hstack((X, b))

        return X 

    def _subgrad(self, x, y, w):
        '''
        Calculate the subgradient.
        '''
        check = y * w.T @ x
        if check < 1:
            return w - self.C * self.N * y * x

        else:
            return w

    def predict(self, X):
        '''
        make predictions with the svm model.
        '''
        # fold in bias term
        X = self._fold_bias(X)
        preds = X @ self.w
        return np.where(preds <= 0, -1, 1)

    def error(self, preds, gold):
        '''
        return the error rate.
        '''
        #filter_array = preds == gold
        #print("filter_array=",filter_array)
        #print("len(preds[filter_array])=",len(preds[filter_array]))
        return np.sum(preds != gold) / len(preds) 
        
class DualSVM():
    def __init__(self, C=100/873, kernel=None, gamma=None):
        self.C = C
        self.w = None
        self.b = None
        self.kernel = kernel
        self.gamma = gamma

    def objective_func(self, alphas, X, y):
        '''
        Compute the dual objective function for SVMs using alpha vector, a training matrix X, and the vector of labels y.
        '''
        double_sum = alphas.T @ ((X @ X.T) * (y @ y.T)) @ alphas
        #print("double_sum=",double_sum)
        return (1/2) * double_sum - np.sum(alphas)
    
    def objective_func_k(self, alphas, kernel, y):
        '''
        Compute the dual objective function for SVMs using alpha vector, a training matrix X, and the vector of labels y.
        '''
        double_sum = alphas.T @ ((kernel) * (y @ y.T)) @ alphas
        #print("double_sum=",double_sum)

        return ((1/2) * double_sum) - np.sum(alphas)

    def _eq_constraint(self, alphas, y):
        return alphas @ y

    def gaussian_kernel(self, x, z, gamma):
        '''
        Calculate the Dot product with Gaussian Kernel. 
        '''
        return np.exp(-np.linalg.norm(x - z, 2, axis=1)**2 / gamma)

    def train(self, X, y):
        '''
        solve the dual svm problem.
        '''
        #args = (X, y)
        args = (X @ X.T, y)
        init_guess = np.zeros(X.shape[0])
        bounds = optimize.Bounds(0, self.C)
        #constraints = optimize.LinearConstraint(y, 0 , 0)
        constraints = [{'type': 'eq', 'fun': self._eq_constraint, 'args': [y]}]
        if self.kernel is not None:
            # this is replacing X^top X from before in the objective
            kernel = np.zeros((X.shape[0], X.shape[0]))
            print("kernel.shape=",kernel.shape)
            # create the gaussian kernel
            for row in range(X.shape[0]):
                new_row = self.gaussian_kernel(X[row,np.newaxis], X, self.gamma)
                #print("new_row=",new_row)
                kernel[row, :] = new_row

            # passing the kernel instead
            args = (kernel, y)

           # print("kernel1 - kernel2=",kernel - kernel2)
        # solve the minimization problem directly
        sol = optimize.minimize(self.objective_func_k, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        
        print("sol=",sol)

        #constraints = [{'type': , 'fun': }]
        alphas = sol.x
        print("alphas=",alphas)
        print("len(alphas)=",len(alphas))

        support_vec_idxs = alphas > 1e-4
        support_vec_alphas = alphas[support_vec_idxs]
        support_vecs = X[support_vec_idxs] 
        support_vec_labels = y[support_vec_idxs]
        print("len(support_vec_labels)=",len(support_vec_labels))
        print("support_vecs=",support_vecs)
        print("support_vec_alphas=",support_vec_alphas)
        print("len(support_vecs)=",len(support_vecs))

        # recover w and b from learned alphas
        w = (alphas * y) @ X
        #w = (support_vec_alphas * support_vec_labels) @ support_vecs
        print("w=",w)

        #if kernel then we don't compute w here. apply kernel in predict

        b = 0
        for vec, label in zip(support_vecs, support_vec_labels):
            b += label - (w.T @ vec)
        
        b = b / len(support_vecs)

        #b = np.mean(support_vec_labels - (support_vec_alphas * support_vec_labels) @ support_vecs * kernel)

        print("b=",b)

        # store learned params
        self.w = w
        self.b = b

    def predict(self, X):
        '''
        make predictions with the svm model.
        '''
        preds = X @ self.w + self.b
        # for non-linear svm just need to apply kernel to X
        return np.where(preds < 0, -1, 1)

    def error(self, preds, gold):
        '''
        return the error rate.
        '''
        return np.sum(preds != gold) / len(preds) 
