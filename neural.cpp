#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class NeuralNetwork;

void test(int epochs, vector<int> hidden_layers, float lr);

int main(){

    test(10000, {16}, 0.1);

    return 0;
}

class SingleExampleNeuralNetwork{
    public:

        //attributes for the SingleExampleNeuralNetwork class

        //inputs from the user
        vector<int> hidden_layers; //vector which gives [no. of nodes in layer1, no. of nodes in layer2,...]
        vector<float> input;
        vector<float> Y_vals;
        vector< vector< vector<float> > > weights;
        vector< vector<float> > biases;
        vector< vector<float> > layers;
        vector<float> output; 
        float lr; //learning rate

        void init_weights_biases(int input_size, int output_size){


            hidden_layers.push_back(output_size); //add output layer size to hidden layer argument 
            hidden_layers.insert(hidden_layers.begin(), input_size); //adds input layer to beginning of hidden_layers


            //initialise random weights biases and hidden layers
            int n = hidden_layers.size() - 1; //number of hidden layers -1 = number of sets of weights and biases

            //for each layer
            for (int i=0;i<n;i++){

                vector< float > bias_layer; //creating one layer of random biases
                vector< vector<float> > weights_layer; //creating one layer of random weights

                //for each node in the upcoming layer
                for (int j=0;j<hidden_layers[i+1];j++){ //move hidden_layers[i] outside of for loop for efficiency
                    
                    bias_layer.push_back((float)(rand() % 100)/1000); //bias value from 0 to 0.1

                    vector< float > weights_single;

                    for (int k=0;k<hidden_layers[i];k++){

                        weights_single.push_back((float)(rand() % 100)/1000); //random weights from 0 to 0.1
                    }
                    weights_layer.push_back(weights_single);

                }
                //create random bias for each layer
                biases.push_back(bias_layer);
                //create random weights for each layer
                weights.push_back(weights_layer);
                
            }
        }


        /*
        vector< vector<float> > weights1; //dimension (size of hidden layer, size of input)
        vector<float> bias1; // size of hidden layer
        vector< vector<float> > weights2; //dimension (size of output, size of hidden layer)
        vector<float> bias2; // size of output
        vector<float> layer1;
        vector<float> output;
        */

        //activation function: sigmoid
        static float sigmoid(float x){
            return 1/(1+exp(-x));
        }

        static float inv_sigmoid(float x){
            return -log(1/x - 1);
        }

        //calculates dot product for application iun feedforward
        static float dot_product(vector<float> x_vec, vector<float> y_vec){

            int n = x_vec.size();
            float total = 0;

            for (int i = 0; i<n; i++){
                total += x_vec[i]*y_vec[i];
            }

            return total;
        }

        //outputs a new layer given the original layer and the weights
        static vector<float> apply_weights(vector<float> layer, vector< vector<float> > cur_weights, vector<float> bias){

            int n = cur_weights.size();

            vector<float> new_layer;

            //for each node in new_layer calculate the dot product of the orignal layer with the weights
            for (int i=0; i<n; i++){

                new_layer.push_back(sigmoid(dot_product(layer, cur_weights[i]) + bias[i]));
            }

            return new_layer;
        }


        void feedforward(){
            int n = hidden_layers.size()-1; //how many times to apply weights and biases (minus 1!)

            vector< vector<float> > test_layers;  //thought this might be needed but gives a segmentation fault when introduced
            test_layers.push_back(input);

            //for all layers in network
            for (int i=0;i<n;i++){

                test_layers.push_back(apply_weights(test_layers[i], weights[i], biases[i])); //for some reason this is not updating layers
            }
            layers = test_layers;
            
            /*
            //calculates hidden layer
            layer1 = apply_weights(input, weights1, bias1);
            //calculates output layer
            output = apply_weights(layer1, weights2, bias2);
            */
        }

        //sigmoid derivative to be applied in update method - soecifically for the partial derivative of the cost function
        static float sigmoid_der(float z){
            return sigmoid(z)*(1-sigmoid(z));
        }

        
        vector<float> update(vector<float> current_layer, vector<float> prev_layer, vector< vector<float> > &current_weights, vector<float> &current_bias, vector<float> Y_true){
            //for each node(j)
            int n = current_layer.size();
            int m = current_weights[0].size();
            vector<float> new_prev_layer(m); //need to calc prev layer with new weights and bias to apply update to each layer


            for (int j=0;j<n;j++){

                //term derived from chain rule
                //for gradient descent all updated parts(bias/weights/neurons) share this term
                float descent = 2*lr*sigmoid_der(inv_sigmoid(current_layer[j]))*(current_layer[j] - Y_true[j]); //maybe have to abs last part

                //update bias
                current_bias[j] = current_bias[j] - descent;

                for (int k=0;k<m;k++){
                    //update weight
                    float new_weight = current_weights[j][k] - prev_layer[k]*descent; //creating this for efficiency as needs to be applied twice
                    current_weights[j][k] = new_weight;

                    //keep track of updated nodes
                    new_prev_layer[k] = prev_layer[k] - new_weight*descent;
                }
            }


            return new_prev_layer; //need to return new_prev_layer to use it again
        }

        void back_prop(){
            int n = hidden_layers.size()-1; //how many times to apply weights and biases (minus 1!)

            vector<float> pre_layer = Y_vals;

            //for all layers in network traversed backward
            for (int i=n;i>0;i--){
                pre_layer = update(layers[i], layers[i-1], weights[i-1], biases[i-1], pre_layer); //update weights and baises traversing the network backward
            }
            /*
            //update final weights and biases, and outputs better prediction for penultimate layer nodes
            vector<float> pre_layer = update(output, layer1, weights2, bias2, Y_vals);
            //update initial weights and biases
            vector<float> pre_pre_layer = update(layer1, input, weights1, bias1, pre_layer);
            */
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
                    cout << i << ' ';
                }
                cout << '\n';
            }

        }
};

class NeuralNetwork: public SingleExampleNeuralNetwork{
    public:
        //attributes for NeuralNetwork class
        vector< vector<float> > all_inputs;
        vector< vector< vector<float> > > all_layers;
        //vector< vector<float> > all_outputs;
        vector< vector<float> > all_Y_vals;

        void train(int epochs){
            //initialise weights and biases
            init_weights_biases(all_inputs[0].size(), all_Y_vals[0].size());
            
            int n = all_inputs.size(); //number of training examples

            //repeat for number of epochs
            for (int j=0;j<epochs;j++){

                //reset after each epoch
                vector< vector< vector<float> > > all_layers;
                vector< vector<float> > all_outputs;


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
            
            }

            //feedforward one last time
            for (int i=0;i<n;i++){
                    input = all_inputs[i];

                    feedforward();

            }
            
        }

        //CANT BE APPLIED CURRENTLY!!
        //this applies the trained weights on an unknown example
        void run_single(vector<float>input_test){
            input = input_test;

            feedforward();

            vector<float> test_output = layers.back();

            int n = test_output.size();
            //for printing purposes
            for (int i=0;i<n;i++){
                cout << test_output[i] << ' ';
            }
        }
    

};


////TESTS////

void test(int epochs, vector<int> hidden_layers, float lr){
    NeuralNetwork neural;

    //attributes to initialise
    neural.lr = lr; //learning rate
    neural.hidden_layers = hidden_layers; //hidden layers[i] = no of nodes in hidden layer i
    neural.all_inputs = {{1.3,-0.6,0.5}, {0.3,1.1,-0.2}, {1.0,1.0,0.4}, {0.9,0.1,-0.3}}; //4 training examples with 3 inputs each
    neural.all_Y_vals = {{0,1},{1,0},{0,1},{1,0}};

    neural.train(epochs);

    neural.run_single({0.4,1.0,-0.2});

}