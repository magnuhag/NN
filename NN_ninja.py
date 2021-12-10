import autograd.numpy as np
from autograd import elementwise_grad as egrad



class Neuralnet:

    def __init__(self):
        
        self.layers = []
        self.act_funcs = [] 
        self.weights = []
        self.biases = []
        self.Z = []
        self.A = []
        self.delta = []
        self.delta2 = []

    def quit(self):
        """lol"""
        return 

    def add(self, n_neurons, act_func, input_size = -1):
        """
        Sequantially adds layer to network in the order (in, hidden_1, ..., hidden_n, out). When adding input layer,
        input size must be specified. 
        """

        if isinstance(n_neurons, int) and n_neurons >= 1:
            self.layers.append(n_neurons)

        else:
            #Should be obvious to anyone attempting to use this class. Still: might catch a typo
            raise TypeError("n_neurons must be of type int and greater than or equal to 1")

        #use -1 as kwarg for input_size, as this is an unlikely value to be added by a user.
        #Might actually change that/ imporve method later
        if input_size != -1 and isinstance(input_size, int):
            self.weights.append(np.random.randn(input_size, n_neurons))
            

        elif input_size == -1:
            self.weights.append(np.random.randn(self.layers[-2], n_neurons))
        #Errrrr
        else:
            raise TypeError("Errr")

        if isinstance(act_func, str):
            self.act_funcs.append(self.activation_function(act_func))
        else:
            raise TypeError("act_func argument must be of type str")

        #Making lists for holding the necessary vectors and matrices
        #Works OK, but not very "pretty"
        self.biases.append(np.random.randn(n_neurons,1))
        self.A.append(0)
        self.Z.append(0)
        self.delta.append(0)
        self.delta2.append(0)

    def activation_function(self, act):
        """
        NOT DOC STRING. 
        Note to self:
        Not sure I'm happy with this method.
        """

        if act == "sigmoid":
            activ = lambda x: 1/(1+np.exp(-x))

        elif act == "RELU":
            activ = lambda x: np.maximum(x, 0)

        elif act == "leaky_RELU":                
            activ = lambda x: np.maximum(x, 0.01 * x)

        elif act == "softmax":

            activ = lambda x: np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True) 

        #Actual name?
        elif act == "no":
            activ = lambda x: x
        
        #Need another solution to this? Also: bad formatting
        else:
            print("-----------------------------------")
            print(" ")
            print(str(act) + " is an invalid activation function name")
            print(" ")
            print("-----------------------------------")

            return
        
        return activ

    def loss_function(self, loss):
        """Under developement. Will be adding several loss functions."""

        if isinstance(loss, str):
            if loss == "MSE":
                func = lambda x, y: np.mean((x - y)**2, axis = 1, keepdims = True)
                return func
            elif loss == "diff":
                func = lambda x, y: x - y
                return func
            else:
                raise ValueError("Invalid loss function name")
        else:
            raise TypeError("Loss function argument must be of type str")        

    def feed_forward(self, X):
        
        if np.shape(X) == (32,6):
            print(np.shape(X))
            print(np.shape(self.weights[0]))

        self.Z[0] = X @ self.weights[0] + self.biases[0].T
        self.A[0] = self.act_funcs[0](self.Z[0])

        for i in range(1, len(self.weights)):

            self.Z[i] = self.A[i-1] @ self.weights[i] + self.biases[i].T
            self.A[i] = self.act_funcs[i](self.Z[i])

        

        return self.A[-1]

    def back_prop(self, y, loss):

            #Will make a method for this soon, allowing for more than one output neuron
            loss_function = self.loss_function(loss)
            #Next line holds a function for later use
            #cross = lambda x, y: -np.sum(y*np.log(x))
            #final layer
            #Taking the derivative of the activation function
            dfdz = egrad(self.act_funcs[-1])
            #Taking the derivative of the loss function w.r.t. to argument "x"
            dcda = egrad(loss_function, 0)

            #Hadamar product of f'(z^L) and dC/da^L
            self.delta[-1] = np.multiply(dfdz(self.Z[-1]), dcda(self.A[-1], y))
            #Differentiating the loss function w.r.t. to the output value
            dcdb = egrad(loss_function, 0)
            #Differentiation the loss function w.r.t. to the bias

            delta_Lj2 = dcda(self.biases[-1].T, y)
            self.delta2[-1] = delta_Lj2

    
            for i in range(len(self.weights)-2, -1, -1):
                dfdz = egrad(self.act_funcs[i])
                dcda = egrad(loss_function, 0)
                t1 =  self.delta[i+1] @ self.weights[i+1].T
                self.delta[i] = np.multiply(t1, dfdz(self.Z[i]))
                # Quick and dirty fix for an issue where dcda(self.biases[i].T, y) does not broadcast
                #Does not always work (why?)
                try:
                    self.delta2[i] = dcda(self.biases[i].T, y)
                except ValueError:
                    self.delta2[i] = dcda(self.biases[i], y)
            
    def optimizer(self, X, eta):
        """For the moment only supports mini-batch SGD. The mini-batch part is supplied by the train method. 
        Other optimizers comping soon.
        Requires argument eta as learning rate (learning rate feature is currently under development). 
        """

        self.weights[0] -= eta * (X.T @ self.delta[0])
        try:
            self.biases[0] -= eta * self.delta2[0].T
        except ValueError:
            self.biases[0] -= eta * self.delta2[0]


        for i in range(1, len(self.weights)):
            self.weights[i] -= eta * (self.A[i-1].T @ self.delta[i])
            try:
                self.biases[i] -= eta * self.delta2[i].T
            except ValueError:
                self.biases[i] -= eta * self.delta2[i]

    def train(self, X, y, epochs, loss):

        """
        Takes input X (feature matrix), y (targets), and epochs (type int).
        This method is currently being improved. Decreasing learning rate is currently being worked on.
        This (learning rate), and mini-batch implementation will be improved and made optional. Other improvements
        are also on the way. 
        """

        for i in range(len(self.weights)):
            print(np.shape(self.biases[i]))
        
        try:
            X @ self.weights[0]
        except ValueError:
            print("Input size "+str(len(self.weights[0])) +" and X-shape "+str(np.shape(X))+" are not compatible")
            return

        data_indices = len(X)
        batch_size = 30
        num_iters = 400

        eta = lambda eta_init, iteration, decay: eta_init/(1+decay*iteration) 
        eta_init = 10**(-2)
        decay = 0.05

        for i in range(1, epochs+1):
            eta1 = eta(eta_init, i, decay)
            for j in range(num_iters):

                chosen_datapoints = np.random.choice(data_indices, size = batch_size, replace = True)
                X_mini = X[chosen_datapoints]

                y_mini = y[chosen_datapoints]
                self.feed_forward(X_mini)
                self.back_prop(y_mini, loss)
                self.optimizer(X_mini, eta1)
            predicted = self.pred(X_mini)
            acc = self.metric(predicted, y_mini, "accuracy")
            print(acc)

        
    def metric(self, y_hat, y, a):

        if a == "accuracy":
            for i in range(len(y)):
                true = np.argmax(y[i])
                pred = np.argmax(y_hat[i])
                s = 0

                if true == pred:
                    s += 1
                else:
                    continue
            return s/len(y_hat)

    def pred(self, X):
        """
        Returns the values from the output neuron
        """
        self.feed_forward(X)
        return self.A[-1]