#include <iostream>
#include <cmath>
#include <vector>
#include "neural.hpp"


class SingleExampleNeuralNetwork{
    public:

        //attributes for the SingleExampleNeuralNetwork class

        //inputs from the user
        std::vector<int> hidden_layers; //vector which gives [no. of nodes in layer1, no. of nodes in layer2,...]
        std::vector<long double> input;
        std::vector<long double> Y_vals;
        std::vector< std::vector< std::vector<long double> > > weights;
        std::vector< std::vector<long double> > biases;
        std::vector< std::vector<long double> > layers;
        std::vector<long double> output; 
        long double lr; //learning rate

        void init_weights_biases(int input_size, int output_size){


            hidden_layers.push_back(output_size); //add output layer size to hidden layer argument 
            hidden_layers.insert(hidden_layers.begin(), input_size); //adds input layer to beginning of hidden_layers


            //initialise random weights biases and hidden layers
            int n = hidden_layers.size() - 1; //number of hidden layers -1 = number of sets of weights and biases

            //for each layer
            for (int i=0;i<n;i++){

                std::vector< long double > bias_layer; //creating one layer of random biases
                std::vector< std::vector<long double> > weights_layer; //creating one layer of random weights

                //for each node in the upcoming layer
                int num_of_neurons = hidden_layers[i+1];
                for (int j=0;j<num_of_neurons;j++){
                    
                    bias_layer.push_back((long double)(rand() % 100)/1000); //bias value from 0 to 0.1

                    std::vector< long double > weights_single;

                    for (int k=0;k<hidden_layers[i];k++){

                        weights_single.push_back((long double)(rand() % 100)/1000); //random weights from 0 to 0.1
                    }
                    weights_layer.push_back(weights_single);

                }
                //create random bias for each layer
                biases.push_back(bias_layer);
                //create random weights for each layer
                weights.push_back(weights_layer);
                
            }
        }

        //activation function: sigmoid
        static long double sigmoid(long double x){
            return 1/(1+exp(-x));
        }

        static long double inv_sigmoid(long double x){
            return -log(1/x - 1);
        }

        //calculates dot product for application in feedforward
        static long double dot_product(std::vector<long double> x_vec, std::vector<long double> y_vec){

            int n = x_vec.size();
            long double total = 0;

            for (int i = 0; i<n; i++){
                total += x_vec[i]*y_vec[i];
            }

            return total;
        }

        //outputs a new layer given the original layer and the weights
        static std::vector<long double> apply_weights(std::vector<long double> layer, std::vector< std::vector<long double> > cur_weights, std::vector<long double> bias){

            int n = cur_weights.size();

            std::vector<long double> new_layer;

            //for each node in new_layer calculate the dot product of the orignal layer with the weights
            for (int i=0; i<n; i++){

                new_layer.push_back(sigmoid(dot_product(layer, cur_weights[i]) + bias[i]));
            }

            return new_layer;
        }


        void feedforward(){
            int n = hidden_layers.size()-1; //how many times to apply weights and biases (minus 1!)

            std::vector< std::vector<long double> > test_layers;
            test_layers.push_back(input);

            //for all layers in network
            for (int i=0;i<n;i++){

                test_layers.push_back(apply_weights(test_layers[i], weights[i], biases[i]));
            }
            layers = test_layers;
            output = layers.back(); //testing
        }

        //sigmoid derivative to be applied in update method - soecifically for the partial derivative of the cost function
        static long double sigmoid_der(long double z){
            return sigmoid(z)*(1-sigmoid(z));
        }

        
        std::vector<long double> update(std::vector<long double> current_layer, std::vector<long double> prev_layer, std::vector< std::vector<long double> > &current_weights, std::vector<long double> &current_bias, std::vector<long double> Y_true){
            //for each node(j)
            int n = current_layer.size();
            int m = current_weights[0].size();
            std::vector<long double> new_prev_layer(m); //need to calc prev layer with new weights and bias to apply update to each layer

            //for each neuron in layer
            for (int j=0;j<n;j++){

                //term derived from chain rule
                //for gradient descent all updated parts(bias/weights/neurons) share this term
                long double descent = 2*lr*sigmoid_der(inv_sigmoid(current_layer[j]))*(current_layer[j] - Y_true[j]); //maybe have to abs last part

                //update bias
                current_bias[j] = current_bias[j] - descent;

                for (int k=0;k<m;k++){
                    //update weight
                    long double new_weight = current_weights[j][k] - prev_layer[k]*descent; //creating this for efficiency as needs to be applied twice
                    current_weights[j][k] = new_weight;

                    //keep track of updated nodes
                    new_prev_layer[k] = prev_layer[k] - new_weight*descent;
                }
            }

            return new_prev_layer; //need to return new_prev_layer to use it again
        }

        void back_prop(){
            int n = hidden_layers.size()-1; //how many times to apply weights and biases (minus 1!)

            std::vector<long double> pre_layer = Y_vals;

            //for all layers in network traversed backward
            for (int i=n;i>0;i--){
                pre_layer = update(layers[i], layers[i-1], weights[i-1], biases[i-1], pre_layer); //update weights and baises traversing the network backward
            }
        }

        void run(int epochs){
            //initialise weights and biases
            init_weights_biases(input.size(), Y_vals.size());

            //initial feedforward
            feedforward();

            //for each epoch, apply back propagation and feedforward again
            for (int i=0;i<epochs;i++){
                back_prop();
                feedforward();
                
                //output for visual aid
                for (auto i: output){
                    std::cout << i << ' ';
                }
                std::cout << '\n';
            }

        }
};

class NeuralNetwork: public SingleExampleNeuralNetwork{
    public:
        //attributes for NeuralNetwork class
        std::vector< std::vector<long double> > all_inputs;
        //std::vector< std::vector< std::vector<long double> > > all_layers;
        //std::vector< std::vector<long double> > all_outputs;
        std::vector< std::vector<long double> > all_Y_vals;

        //this training method applies the feedforward to each training example before aplpying the back propagation
        void train(int epochs){
            //initialise weights and biases
            init_weights_biases(all_inputs[0].size(), all_Y_vals[0].size());
            
            int n = all_inputs.size(); //number of training examples

            //repeat for number of epochs
            for (int j=0;j<epochs;j++){

                lr = (lr-0.05)*(epochs-j)/epochs + 0.05;

                //reset after each epoch
                std::vector< std::vector< std::vector<long double> > > all_layers;
                //std::vector< std::vector<long double> > all_outputs;


                // for each training example, apply feedforward
                for (int i=0;i<n;i++){
                    input = all_inputs[i];

                    feedforward();

                    //update layers for back_prop
                    all_layers.push_back(layers);

                }

                // for each training example, apply back propagation
                for (int i=0;i<n;i++){
                    input = all_inputs[i];
                    Y_vals = all_Y_vals[i];
                    layers = all_layers[i];

                    back_prop();
                    
                }


                std::cout << j << "\n"; //for visual purposes

            
            }

            //is the below actually needed
            /*
            //feedforward one last time
            for (int i=0;i<n;i++){
                    input = all_inputs[i];

                    feedforward();

            }
            */
            
        }

        //this training method applies the feedforward and backprop to each training example before moving to the next one
        void train2(int epochs){
            //initialise weights and biases
            init_weights_biases(all_inputs[0].size(), all_Y_vals[0].size());
            
            int n = all_inputs.size(); //number of training examples

            //repeat for number of epochs
            for (int j=0;j<epochs;j++){
                
                //reset after each epoch
                std::vector< std::vector< std::vector<long double> > > all_layers;
                //std::vector< std::vector<long double> > all_outputs;


                // for each training example, apply feedforward
                for (int i=0;i<n;i++){
                    input = all_inputs[i];

                    feedforward();

                    input = all_inputs[i];
                    Y_vals = all_Y_vals[i];

                    back_prop();
                    
                }

                std::cout << "*\n";
            
            }

            //feedforward one last time
            for (int i=0;i<n;i++){
                    input = all_inputs[i];

                    feedforward();

            }
            
        }

        // This is NOT working as it should at the moment
        void test(std::vector< std::vector<long double> > X_test, std::vector< std::vector<long double> > y_test){
            int n = X_test.size(); //number of examples in test set

            for (int i=0;i<n;i++){
                input = X_test[i];

                feedforward();

                std::vector<long double> test_output = layers.back();

                int m = test_output.size();
                //for printing purposes
                for (int j=0;j<m;j++){
                    std::cout << test_output[j] << ' ';
                }
                std::cout << '\n';
                //printing true value
                for (int j=0;j<m;j++){
                    std::cout << y_test[i][j] << ' ';
                }
                std::cout << '\n';
            }
        }

        //this applies the trained weights on an unknown example
        void run_single(std::vector<long double>input_test){
            input = input_test;


            feedforward();

            //std::vector<long double> test_output = layers.back();

            //int n = test_output.size();
            int n = output.size();


            //for printing purposes
            for (int i=0;i<n;i++){
                //std::cout << test_output[i] << ' ';
                
                std::cout << output[i] << ' ';
            }
            std::cout << '\n';
        }
    
};