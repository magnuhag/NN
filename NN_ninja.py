import autograd.numpy as np
from autograd import elementwise_grad as egrad



class NeuralNet:
    def __init__(self):
        """Feed forward neural network. Currently under development, and thus is lacks several key features. 
        Currently only works one output neuron, but this will be fixed ASAP. I've left a lot of notes to my self scattered around  in the code.
        Please excuse these. List (in order of importance) of lacking features and other stuff that needs improvement:
        
        -Make a (working) loss function method compatible with an arbitrary number of output neurons
        -Make the back_prop method compatible with the forthcoming loss_function method
        -General improvements to the optimizer and train methods. Currently there are features in one the should be in the other, etc.
        -General improvements to the activation_function method
        -Possibly combine the add and create methods 
        -Possibly add a method for evaluating the predictions made by the network.
        -Clean up the code, consolidate features, make general improvements as improvements are needed, and remove unnecessary comments/ make better comments
        """
        self.shapes = []
        self.activation = []

    def activation_function(self, act):
        """Activation function for layer. Argument must be of type str. Currently the avalable 
        activations are "sigmoid", "RELU", "leaky_RELU", and "softmax". If anything other than these four is input, 
        the activation function is y(x)=x (commonly used as output in regression)."""

        #This method must be imporved. Not prioritized. 
        
        if act == "sigmoid":
            activation = lambda x: 1/(1+np.exp(-x))
            d_activation = lambda x: np.exp(-x)/(1+np.exp(-x))**2
 
        elif act == "RELU":
            activation = lambda x: np.maximum(x, 0)
            def d_activation(x):
                x[x<=0] = 0
                x[x>0] = 1
                return x

        elif act == "leaky_RELU":
            activation = lambda x: np.maximum(x, 0.01 * x)
            def d_act(x):
                alpha = 0.01
                dx = np.ones_like(x)
                dx[x < 0] = alpha
                return dx


        elif act == "softmax":
            exp_term = lambda x: np.exp(x)
            activation = lambda x: exp_term/np.sum(exp_term, axis = 1, keepdims = True) 

        else:
            activation = lambda x: x

        return activation

    def loss_function(self, loss):
        """Under developement. Will be adding several loss functions, and make the method play nice with the back_prop and train methods"""

        if loss == "MSE":
            func = lambda x, y: np.mean((x - y)**2, axis = 1, keepdims = True)
            return func
        elif loss == "diff":
            func = lambda x, y: x - y
            return func

    def add(self, shape, act):
        """Adds a layer to network. These are added sequentially (input, first, second, ..., output). 
        The Shape argument assigns number of neurons to layer and must be of type
        int. act is which activation function is to be used. 
        First layer must have same number of neurons as number of features in feature matrix.
        More info on available activation functions can be accessed in the activation_function method of this class""" 
        self.shapes.append(shape)
        activation = self.activation_function(act)
        self.activation.append(activation)

    def create(self):
        """Creates the (not really, mathematically speaking) "tensors" containing the weights and biases"""

        #Not sure if this should be a seperate method. Perhaps change the shape of self.weights[i] to (self.shapes[i], self.shapes[i+1])
        #so that it is in congruence with the equations. Doesn't really matter though

        for i in range(len(self.shapes)-1):
            weight_matrix = np.random.randn(self.shapes[i+1], self.shapes[i])*0.1
            self.weights[i] = weight_matrix
            bias_vector = np.random.randn(len(self.weights[i][:]), 1)*0.1
            self.biases[i] = bias_vector

    def feed_forward(self, X):
        """Takes the feature matrix (or vector or whatever) and feeds it throught the network. 
        Arument X must be numpy.ndarray"""
        self.compile()
        self.create()
        #Feeding feature matrix into first layer
        Z = self.weights[0] @ X.T + self.biases[0]

        #Output of first layer
        A = self.activation[0](Z)
        self.Z[0] = Z
        self.A[0] = A


        for i in range(1, len(self.weights)):  
            #Feed forward to layer i
            Z = self.weights[i] @ self.A[i-1] + self.biases[i]

            A = self.activation[i](Z)
            self.Z[i] = Z
            self.A[i] = A

        #Remember to delete this return
        return self.A[-1]

    def back_prop(self, y, loss):

        """Backpropegating."""

        #Will make a method for this soon, allowing for more than one output neuron
        loss_function = self.loss_function(loss)
        #Next line holds a function for later use
        #cross = lambda x, y: -np.sum(y*np.log(x))
        #final layer
        #Taking the derivative of the activation function
        dfdz = egrad(self.activation[-1])
        #Taking the derivative of the loss function w.r.t. to argument "x", 
        dcda = egrad(loss_function, 0)

        #Hadamar product of f'(z^L) and dC/da^L
        print(np.shape(dcda(self.A[-1], y)))

        print(np.shape(dfdz(self.Z[-1])), "dfdz")
        delta_Lj = np.multiply(dfdz(self.Z[-1]), dcda(self.A[-1], y))
        self.delta[-1] = delta_Lj
        #Differentiating the loss function w.r.t. to the output value
        dcdb = egrad(loss_function, 0)
        #Differentiation the loss function w.r.t. to the bias
        delta_Lj2 = dcda(self.biases[-1], y)
        self.delta2[-1] = delta_Lj2
  
        for i in range(len(self.weights)-2, -1, -1):
            
            dfdz = egrad(self.activation[i])
            dcda = egrad(loss_function, 0)
            t1 = self.weights[i+1].T @ self.delta[i+1]

            self.delta[i] = np.multiply(t1, dfdz(self.Z[i]))
            self.delta2[i] = dcda(self.biases[i], y)


    def optimizer(self, eta):
        """For the moment only supports mini-batch SGD. The mini-batch part is supplied by the train method. 
        Other optimizers comping soon.
        Requires argument eta as learning rate (learning rate feature is currently under development). 
        """

        for i in range(len(self.weights)):
            self.weights[i] -= eta * (self.delta[i] @ self.A[i-1].T)
            self.biases[i] -= eta * self.delta2[i]

    def compile(self):
        """Builds lists to contain vectors and arrays"""

        #Want all of these to be lists of length self.shapes, so they can contain the
        #appropriate number of arbirarily dimensioned arrays
        self.A = [0 for i in range(len(self.shapes)-1)]
        self.Z = self.A.copy()
        self.weights = self.A.copy()
        self.biases = self.A.copy()
        self.delta = self.A.copy()
        self.delta2 = self.A.copy()

    def train(self, X, y, epochs, loss):

        """
        Takes input X (feature matrix), y (targets), and epochs (type int).
        This method is currently being improved. Decreasing learning rate is currently being worked on.
        This (learning rate), and mini-batch implementation will be improved and made optional. Other improvements
        are also on the way. 
        """


        self.compile()
        self.create()

        data_indices = len(X)
        batch_size = 30
        num_iters = 400

        eta = lambda eta_init, iteration, decay: eta_init/(1+decay*iteration) 
        eta_init = 1
        decay = 0.01

        for i in range(epochs):
            for j in range(num_iters):

                chosen_datapoints = np.random.choice(data_indices, size = batch_size, replace = False)
                X_mini = X[chosen_datapoints]

                y_mini = y[chosen_datapoints]
                self.feed_forward(X_mini)
                self.back_prop(y_mini, loss)
                self.optimizer(eta(eta_init, i, decay))
                asn = self.pred(X)
            print(np.mean((y-asn)**2), "error at epoch" + str(i))
    def pred(self, X):
        """
        Returns the values from the output neuron
        """
        self.feed_forward(X)
        return self.A[-1]
