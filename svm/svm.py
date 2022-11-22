# This program implements the SVM class
import numpy as np
#from scipy import optimize
import cvxopt

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


# Kernel Functions
def gaussian_kernel(x, z, gamma):
    '''
    Calculate the Dot product with Gaussian Kernel. 
    '''
    return np.exp(-np.linalg.norm(x - z, 2, axis=1)**2 / gamma)

def linear_kernel(x, z, gamma=None):
    '''
    Calculate the linear kernal.
    '''
    return x @ z.T
        
class DualSVM():
    def __init__(self, C=100/873, kernel=None, gamma=None):
        self.C = C
        self.w = None
        self.b = None
        self.kernel = kernel

        if self.kernel == 'linear':
            self.kernel_func = linear_kernel
        if self.kernel == 'gaussian':
            self.kernel_func = gaussian_kernel
        self.gamma = gamma


    def objective_func_k(self, alphas, kernel, y):
        '''
        Compute the dual objective function for SVMs using alpha vector, a training matrix X, and the vector of labels y.
        '''
        double_sum = alphas.T @ ((kernel) * (y @ y.T)) @ alphas
        #print("double_sum=",double_sum)

        return ((1/2) * double_sum) - np.sum(alphas)

    def _eq_constraint(self, alphas, y):
        return alphas @ y


    def train(self, X, y):
        '''
        solve the dual svm problem.
        '''
        self.X = X
        self.y = y
        self.num_examples = X.shape[0]

        kernel = np.zeros((X.shape[0], X.shape[0]))
        #print("kernel.shape=",kernel.shape)
        # create the gaussian kernel
        for row in range(X.shape[0]):
            new_row = self.kernel_func(X[row,np.newaxis], X, self.gamma)
            #print("new_row=",new_row)
            kernel[row, :] = new_row

        self.kernel_mat = kernel

        # Optimization Tool - have to format in a certain way.
        P = cvxopt.matrix(np.outer(y, y) * self.kernel_mat)
        q = cvxopt.matrix(-np.ones((self.num_examples, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(self.num_examples) * -1, np.eye(self.num_examples))))
        h = cvxopt.matrix(np.hstack((np.zeros(self.num_examples), np.ones(self.num_examples) * self.C)))
        A = cvxopt.matrix(y, (1, self.num_examples), "d")
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"]).flatten()
        
        #print("self.alphas=",self.alphas)

    def recover_weights(self):
        '''
        use the learned alpha values to recover w and b.
        '''
        
        self.alphas[np.isclose(self.alphas, 0)] = 0  # zero out nearly zeros
        self.alphas[np.isclose(self.alphas, self.C)] = self.C  # round the ones that are nearly C

        #print("self.alphas=",self.alphas)
        #print("len(self.alphas)=",len(self.alphas))

        sv_idxs = self.alphas > 0
        sv_alphas = self.alphas[sv_idxs]
        svs = self.X[sv_idxs] 
        sv_labels = self.y[sv_idxs]
        #print("len(support_vec_labels)=",len(sv_labels))
        #print("support_vecs=",svs)
        #print("support_vec_alphas=",sv_alphas)
        #print("len(support_vecs)=",len(svs))

        # recover w and b from learned alphas
        #w = (self.alphas * y) @ X
        w = (sv_alphas * sv_labels) @ svs
        #print("w=",w)

        b = np.mean(sv_labels - (sv_alphas * sv_labels) * self.kernel_mat[sv_idxs, sv_idxs])

        #print("b=",b)

        # store learned params
        self.w = w
        self.b = b
        self.sv_idxs = sv_idxs
        self.svs = svs
        return sv_idxs

    def predict(self, X):
        '''
        make predictions with the svm model.
        '''
        sv_idxs = self.recover_weights()

        #print("self.alphas[sv_idxs]=",self.alphas[sv_idxs])
        #print("self.y[sv_idxs]=",self.y[sv_idxs])

        preds = np.zeros(X.shape[0])
        for i in range(len(preds)):
            preds[i] = np.sum(self.alphas[sv_idxs] * self.y[sv_idxs, np.newaxis] * self.kernel_func(X[i], self.X[sv_idxs], self.gamma)[:, np.newaxis])

        #print("preds=",preds)
        return np.where(preds + self.b < 0, -1, 1)

    def error(self, preds, gold):
        '''
        return the error rate.
        '''
        return np.sum(preds != gold) / len(preds) 
