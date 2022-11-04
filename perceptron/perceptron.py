# This program implements the Perceptron Class
import numpy as np

class Standard_Perceptron:
    def __init__(self, epochs=10, lr=.1):
        '''
        Initialize weights and bias of a standard Perceptron.
        '''
        self.w = None
        self.epochs = epochs
        self.lr = lr

    def _predict_all(self, X, w):
        '''
        Compute predictions for all instances in matrix X.
        '''
        preds = X @ w 
        return np.where(preds <= 0, -1, 1) 


    def _fold_bias(self, X):
        '''
        Add new element with value of 1 to beginning of each training instance to represent the bias.
        '''

        b = np.ones((X.shape[0], 1))

        # fold in bias term: don't use b anymore
        X = np.hstack((b, X))

        return X 

    def train(self, X, y):
        '''
        Run the training loop.
        '''
        #np.random.seed(47)
        # modify labels to be in {-1, 1} instead of {0,1}
        y = np.copy(y)
        y[y == 0] = -1

        # initialize the weight vector to match input dims
        # add an extra dim for the bias term.
        num_dims = X.shape[1]
        self.w = np.zeros(num_dims + 1)


        # fold in bias term to w and X
        X = self._fold_bias(X)

        for _ in range(self.epochs):
            #np.random.shuffle(X)
            # update loop
            for sample, y_i in zip(X, y):
                if y_i * self.w.T @ sample <= 0:
                    self.w = self.w + self.lr * (y_i * sample)
        # return final weight vector
        return self.w

    def predict(self, X):
        '''
        Make a prediction with the model.
        '''
        # fold in bias term to testing data.
        X = self._fold_bias(X)

        # predict all instances using self.w
        preds = self._predict_all(X, self.w)
        return preds


class Voted_Perceptron(Standard_Perceptron):
    def __init__(self, epochs, lr):
        super().__init__(epochs=epochs, lr=lr)
        # track history of  
        self.w_hist = []
        self.counts = []

    def train(self, X, y):
        '''
        Run the training loop.
        '''
        #np.random.seed(47)
        # modify labels to be in {-1, 1} instead of {0,1}
        y = np.copy(y)
        y[y == 0] = -1

        # initialize the weight vector to match input dims
        # add an extra dim for the bias term.
        num_dims = X.shape[1]
        self.w = np.zeros(num_dims + 1)


        # fold in bias term to w and X
        X = self._fold_bias(X)

        # In this loop we keep track of all of the learned weight vectors
        # And their respective counts
        c = 0
        for e in range(self.epochs):
            #np.random.shuffle(X)
            # update loop
            for sample, y_i in zip(X, y):
                if y_i * self.w.T @ sample <= 0:
                    # append w to the history
                    self.w_hist.append(self.w)
                    self.counts.append(c)
                    # update the weight vector (think of as a new classifier)
                    self.w = self.w + self.lr * (y_i * sample)

                    # reset counter
                    c = 1
                else:
                    c += 1

        # add last weight vector and count
        self.w_hist.append(self.w)
        self.counts.append(c)
        return self.w_hist, self.counts

    def _predict_all(self, X, w_hist, counts):
        '''
        Compute weighted predictions for all instances in matrix X.
        '''
        all_preds = []
        for w,c in zip(w_hist, counts):
            preds = c * X @ w 
            preds = np.where(preds <= 0, -1, 1)
            #print("preds.shape=",preds.shape)
            all_preds.append(preds)
        
        # convert to numpy array then sum
        all_preds = np.array(all_preds)
        #print("all_preds.shape  =",all_preds.shape  )
        res = np.sum(all_preds, axis=0) / len(w_hist)
        #print("res.shape=",res.shape)
        
        return np.where(preds <= 0, -1, 1) 

    def predict(self, X):
        '''
        Make a prediction with the model.
        '''
        # fold in bias term to testing data.
        X = self._fold_bias(X)

        # predict all instances using self.w
        preds = self._predict_all(X, self.w_hist, self.counts)

        return preds

class Averaged_Perceptron(Standard_Perceptron):
    def __init__(self, epochs, lr):
        super().__init__(epochs=epochs, lr=lr)
        self.a = None

    def train(self, X, y):
        '''
        Run the training loop.
        '''
        #np.random.seed(47)
        # modify labels to be in {-1, 1} instead of {0,1}
        y = np.copy(y)
        y[y == 0] = -1

        # initialize the weight vector to match input dims
        # add an extra dim for the bias term.
        num_dims = X.shape[1]
        self.w = np.zeros(num_dims + 1)

        # initialize the 'a' vector
        self.a = np.zeros(num_dims + 1)


        # fold in bias term to w and X
        X = self._fold_bias(X)

        for _ in range(self.epochs):
            #np.random.shuffle(X)
            # update loop
            for sample, y_i in zip(X, y):
                if y_i * self.w.T @ sample <= 0:
                    self.w = self.w + self.lr * (y_i * sample)
                
                # adding the current weight vector to a each iter
                self.a += self.w
        # return final 'a' vector
        return self.a

    def predict(self, X):
        '''
        Make a prediction with the model.
        '''
        # fold in bias term to testing data.
        X = self._fold_bias(X)

        # predict all instances using self.a
        preds = self._predict_all(X, self.a)
        return preds
