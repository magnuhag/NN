import autograd.numpy as np
from autograd import jacobian 
from autograd import elementwise_grad as egrad


class NeuralNet:

    def __init__(self):
        
        #Lists for holding the weight, bias, etc, matrices/ vectors.
        #Call them (for the time being) "empty" tensors, if you're so inclined
        self.layers = []
        self.act_funcs = [] 
        self.d_act_funcs = []
        self.weights = []
        self.biases = []
        self.Z = []
        self.A = []
        self.delta = []
    def add(self, n_neurons, act_func, input_size = None):
        """
        Sequantially adds layer to network in the order (in, hidden_1, ..., hidden_n, out). When adding input layer,
        input size must be specified. 
        """

        if isinstance(n_neurons, int) and n_neurons >= 1:
            self.layers.append(n_neurons)

        else:
            #Should be obvious to anyone attempting to use this class. Still: might catch a typo
            raise TypeError("n_neurons must be of type int and greater than or equal to 1")

        if isinstance(input_size, int):
            self.weights.append(np.random.randn(input_size, n_neurons)*0.01)
            

        elif isinstance(input_size, type(None)):
            self.weights.append(np.random.randn(self.layers[-2], n_neurons)*0.01)
        #Errrrr
        else:
            raise TypeError("Errr")

        if isinstance(act_func, str):
            function = self.activation_function(act_func)
            self.act_funcs.append(function)
        else:
            raise TypeError("act_func argument must be of type str")

        #Making lists for holding the necessary vectors and matrices
        #Works OK, but not very "pretty"
        self.biases.append(np.random.randn(n_neurons)*0.01)
        self.A.append(0)
        self.Z.append(0)
        self.delta.append(0)

    def activation_function(self, act):
        """
        NOT DOC STRING. 
        Note to self:
        Not happy with this method.
        """

        if act == "sigmoid":
            def activ(x):
                return 1/(1+np.exp(-x))

        elif act == "RELU":
            def activ(x):
                return np.maximum(x, 0)

        elif act == "leaky_RELU": 
            def activ(x):
                return np.maximum(x, 0.01 * x)               

        elif act == "softmax":
            def activ(x):
                return np.exp(x)/np.sum(np.exp(x))

        elif act == "linear":
            def activ(x):
                return x
        
        #Yes, formatting
        else:
            print("-----------------------------------")
            print(" ")
            print(str(act) + " is an invalid activation function name")
            print(" ")
            print("-----------------------------------")

            return
        
        return activ     

    def loss_function(self, loss):
        """Under developement. Will be adding more loss functions."""

        if isinstance(loss, str):
            if loss == "MSE":
                def func(x, y):
                    return np.mean((x - y)**2, axis = 1, keepdims = True)
                
            elif loss == "categorical_cross":
                def func(x, y):
                    return -np.sum(y*np.log(x), axis = 1)
                
            else:
                raise ValueError("Invalid loss function name")
                
        else:
            raise TypeError("Loss function argument must be of type str")  
            
        return func
        
    def feed_forward(self, X):
        
        #Feeding in feature matrix
        self.Z[0] = X @ self.weights[0] + self.biases[0].T
        #Activation in first hidden layer
        self.A[0] = self.act_funcs[0](self.Z[0])

        for i in range(1, len(self.weights)):
            #Feeding forward
            self.Z[i] = self.A[i-1] @ self.weights[i] + self.biases[i].T
            self.A[i] = self.act_funcs[i](self.Z[i])

    def diff(self, C, A):
        """
        Not sure this method is of any real use
        """
        dCda = egrad(C)
        dAdz = jacobian(A)

        return dCda, dAdz

    def back_prop(self, y, diff):
            
            #Assigning Jacobian and derivative functions as variables
            dC, da = diff
            #"Empty" (Zeros) array to hold Jacobian
            d_act = np.zeros(len(self.Z[-1]))
            #Empty array to hold derivative of cost function
            dcda = d_act.copy()
            #Empty array to hold delta^L
            self.delta[-1] = np.zeros((len(self.Z[-1]), self.layers[-1]))
            for i in range(len(self.Z[-1])):
                #Calculate Jacobian and derivative for each training example in batch. Unnecessary if batchsize = 1
                d_act = da(self.Z[-1][i])
                dcda = dC(self.A[-1][i], y)
                #Jacobian of activation times derivative of cost function (Hadamard product)
                self.delta[-1][i] = d_act @ dcda
            
            for i in range(len(self.weights)-2, -1, -1):
                #Gradient of activation function of hidden layer i. No need for Jacobian here
                dfdz = egrad(self.act_funcs[i])
                #Equation 2 is calculated in 2 parts. Just for ease of reading
                t1 =  self.delta[i+1] @ self.weights[i+1].T
                self.delta[i] = np.multiply(t1, dfdz(self.Z[i]))
      
    def optimizer(self, X, eta):
        """
        For the moment only supports mini-batch SGD. More will come (maybe)
        """

        self.weights[0] -= eta * (X.T @ self.delta[0])
        self.biases[0] -= eta * np.sum(self.delta[0], axis = 0)

        for i in range(1, len(self.weights)):
            self.weights[i] -= eta * (self.A[i-1].T @ self.delta[i])
            self.biases[i] -= eta * np.sum(self.delta[i], axis = 0)

    def train(self, X, y, epochs, loss, metric, batch_size = 10, num_iters = 100, eta_init = 10**(-4), decay = 0.1):

        """
        Takes args: X (feature matrix), y (targets), and epochs (type int).
        Takes kwargs: batch_size, num_iters, eta_init, decay. The "standard" values provided by the method
        has been found by testing on one dataset. You should probably not use the values I´ve found
        """
        
        diff = self.diff(self.loss_function(loss), self.act_funcs[-1])
        
        data_indices = len(X)
        #eta function (not the Dirichlet one): for decreasing learning rate as training progresses
        eta = lambda eta_init, iteration, decay: eta_init/(1+decay*iteration) 

        for i in range(1, epochs+1):
            
            for j in range(num_iters):
                eta1 = eta(eta_init, j, decay)
                #Randomly choose datapoints to use as mini-batches
                chosen_datapoints = np.random.choice(data_indices, size = batch_size, replace = True)
                #Making mini-batches
                X_mini = X[chosen_datapoints]
                y_mini = y[chosen_datapoints]
                #Feed forward
                self.feed_forward(X_mini)
                #Backprop
                self.back_prop(y_mini, diff)
                #Update weights and biases
                self.optimizer(X_mini, eta(eta_init, j, decay))
                
            #Make a prediction and print mean of performance of mini-batch 
            predicted = self.predict(X_mini)
            performance = self.metrics(predicted, y_mini, metric)
            print(metric +" is " + str(np.mean(performance))+ " at epcoh " +str(i))


    def metrics(self, y_hat, y, a):
        """
        Takes args: y_hat, y, a (prediction, targets, activation in layer L)
        """
        if a == "accuracy":
            s = 0
            for i in range(len(y)):
                true = np.argmax(y[i])
                pred = np.argmax(y_hat[i])
                if true == pred:
                    s += 1
                else:
                    continue

            return s/len(y_hat)

        elif a == "MSE":
            return np.mean((y-y_hat)**2, axis = 0)


    def predict(self, X):
        """
        Takes arg: X
        Does one feed forward pass and returns the output of last layer
        """
        self.feed_forward(X)
        return self.A[-1]