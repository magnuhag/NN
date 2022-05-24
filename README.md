# Hobby project project: building a neural network 

I wrote a neural network as an assignment/ project whilst doing my degree in physics at the University of Oslo.
I've recently found time to improve upon the original architecture and design of the network, and this is currently what's happening in this repo.
The neural net found here is not complete, but it is currently functional.

At the time the network uses Autograd to compute gradients and Jacobians to perform backpropagation. This comes at a significant computational cost, but spares us taking derivatives of cost functions and activation functions our selves. 

I´ve included two files in this repo (in addition to this README). The first one NN_ninja.py is just the code for the network. The other file is a Jupyter Notebook I wrote to mathematically explain what is going on in NN_ninja.py. 
