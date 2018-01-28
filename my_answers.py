
#Implement the sigmoid function to use as the activation function. Set self.activation_function in __init__ to your sigmoid function.
#Implement the forward pass in the train method.
#Implement the backpropagation algorithm in the train method, including calculating the output error.
#Implement the forward pass in the run method.

#Following the struct show in the class, I just re-organize the struct from my_answers.py

#First we need to import the numpy
import numpy as np

#Here it's create the NeuralNetwork
class NeuralNetwork(object):
    #Here we properly initialize the neural network
    #During the construction I put some 'prints' to help me unsderstand the data and how to shape the variables
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        #print('self.input_nodes',self.input_nodes)
        self.hidden_nodes = hidden_nodes
        #print('self.hidden_nodes',self.hidden_nodes)
        self.output_nodes = output_nodes
        #print('self.output_nodes',self.output_nodes)
        #type(self.output_nodes)
        #print(self.output_nodes)
        # Initialize weights first 
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, \
                                                    (self.input_nodes, self.hidden_nodes))
        
        #print('self.weights_input_to_hidden',self.weights_input_to_hidden)
        #type(self.weights_input_to_hidden)
        #print(self.weights_input_to_hidden)


        #Weights output = [NX1]
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, \
                                                     (self.hidden_nodes, self.output_nodes))

        #print('self.weights_hidden_to_output',self.weights_hidden_to_output)
    
        self.lr = learning_rate
        #print('self.lr',self.lr)
    
        #### Set self.activation_function to sigmoid function ####
        # self.activation_function = lambda x : 1/(1+ np.exp(-x))
        #The output node has a 1 function actvation  
    
        def sigmoid(x):
            return 1/(1+np.exp(-x))  
    
        self.activation_function = sigmoid
        #print('self.activation_function',self.activation_function

    def train(self, features, targets):
        # Convert inputs list to 2D array: Input is 56X1 representing one row of inputs 
        inputs = np.array(features, ndmin=2)
        features.shape
        inputs.shape
        #print('feature in trains is',features)
        #print('target in train is',targets)
        #print('input in train is', inputs)        
    
        # n_records is the first index in features.shape: 16,875
        n_records = features.shape[0]
        #print('n_records in train is',n_records)
    
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        #print('delta_weights_i_h in train is',delta_weights_i_h)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        #print('delta_weights_h_o in train is',delta_weights_h_o)
    
        for X, y in zip(features, targets):
        
            ### FORWARD PASS ###
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            #print('hidden_inputs in train is',hidden_inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
            #print('hidden_outputs in train is',hidden_outputs) 

            # Output layer
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            #print('final_inputs in train is',final_inputs)
            final_outputs = final_inputs 
            #print('final_outputs in train is',final_outputs)
        
            ### BACKWARD PASS ###
            output_error = y - final_outputs
            #print('output_error in train is',output_error)
            output_grad = 1
        
            # Backpropagated output_error term 
            output_error_term = output_error * output_grad
      
            hidden_error = np.dot(self.weights_hidden_to_output, output_error)
            #print('hidden_error in train is',hidden_error)

            hidden_error_term = hidden_error * hidden_outputs * (1- hidden_outputs)
            #print('hidden_error_term in train is',hidden_error_term)

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:, None]
            #print('delta_weights_i_h in train is',delta_weights_i_h)

            # Weight step (hidden to output)      
            delta_weights_h_o += output_error_term * hidden_outputs [:, None]
            #print('delta_weights_i_h in train is',delta_weights_i_h)

        # Update the weights 
        # Update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records 
        #print('self.weights_hidden_to_output correct in trains is',self.weights_hidden_to_output)
    
        # Update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
        #print('self.weights_input_to_hidden correct in train is',self.weights_input_to_hidden)
    
    def run(self, features):
        # Run a forward pass through the network
   
        # Signals into hidden layer 
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        #print('hidden_inputs in run is',hidden_inputs)
    
        # Signals into hidden layer 
        hidden_outputs = self.activation_function(hidden_inputs)
        #print('hidden_outputs in run is',hidden_outputs)
    
        # Output layer 
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        #print('final_inputs in run is',final_inputs)
        final_outputs = final_inputs
    
        return final_outputs

#########################################################
# Set your hyperparameters here
##########################################################
iterations = 7000
learning_rate = 0.25
hidden_nodes = 45
output_nodes = 2

#### During the project I insert some prints into the my_awnser.py to help understand the data and the shapes.
#### We can see some behaviors in the Neuralnetworks during the steps to achieve the rubrics, for exemple: 
#### I started with iteratitons 5000, lr 0.50, hidden nodes 30 anda output nodes 1.
#### Decreasing the number of iterations the training loss and the validation loss increase.
#### Decreasing the number of hidden nodes the training loss and the validation loss increase.
#### Increasing the number of learning rate the training loss and the validation loss increase.
#### Finally with the number was perfect achieve with the number in the set hyperparameters.
#### The first step of the NeuralNetwork with low numbers like 100 iterations, lr 1 and 2 hidden nodes, the output prediction is almost a average of the real data.
#### The prediction was really good, really close to the data showing the power of a simple Neuralnetwork.
#### The predict got above the real in almost data but not so much.
#### The time to run all the project is acceptable.

