#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class NeuralNetwork;

void test_feedforward();
void test_back_prop();
void test1(int epochs);
void test2(int epochs);

int main(){

    test2(10000);

    return 0;
}

class SingleExampleNeuralNetwork{
    public:

        //attributes for the SingleExampleNeuralNetwork class
        vector<float> input; 
        vector< vector<float> > weights1; //dimension (size of hidden layer, size of input)
        vector<float> bias1; // size of hidden layer
        vector< vector<float> > weights2; //dimension (size of output, size of hidden layer)
        vector<float> bias2; // size of output
        vector<float> layer1;
        vector<float> output; 
        vector<float> Y_vals;
        float lr = 0.2; //learning rate could be an attribute to class object

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
        static vector<float> apply_weights(vector<float> layer, vector< vector<float> > weights, vector<float> bias){
    
            int n = weights.size();

            vector<float> new_layer;

            //for each node in new_layer calculate the dot product of the orignal layer with the weights
            for (int i=0; i<n; i++){
                new_layer.push_back(sigmoid(dot_product(layer, weights[i]) + bias[i]));
            }

            return new_layer;
        }


        void feedforward(){

            //calculates hidden layer
            layer1 = apply_weights(input, weights1, bias1);
            //calculates output layer
            output = apply_weights(layer1, weights2, bias2);
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

            //update final weights and biases, and outputs better prediction for penultimate layer nodes
            vector<float> pre_layer = update(output, layer1, weights2, bias2, Y_vals);
            //update initial weights and biases
            vector<float> pre_pre_layer = update(layer1, input, weights1, bias1, pre_layer);
        }

        void run(int epochs){
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
        vector< vector<float> > all_layer1;
        vector< vector<float> > all_outputs;
        vector< vector<float> > all_Y_vals;

        void train(int epochs){
            
            int n = all_inputs.size(); //number of training examples

            //repeat for number of epochs
            for (int j=0;j<epochs;j++){

                //reset after each epoch
                vector< vector<float> > all_layer1;
                vector< vector<float> > all_outputs;


                // for each training example, apply feedforward
                for (int i=0;i<n;i++){
                    input = all_inputs[i];

                    feedforward();

                    //update layer1 for back_prop
                    all_layer1.push_back(layer1);
                    //update output for back_prop
                    all_outputs.push_back(output);

                }

                // for each training example, apply back propagation
                for (int i=0;i<n;i++){
                    input = all_inputs[i];
                    Y_vals = all_Y_vals[i];
                    layer1 = all_layer1[i];
                    output = all_outputs[i];

                    back_prop();
                    
                }
            
            }

            //feedforward one last time
            for (int i=0;i<n;i++){
                    input = all_inputs[i];

                    feedforward();
                    all_outputs[i] = output;
            }
            
        }

        //this applies the trained weights on an unknown example
        void run_single(vector<float>input_test){
            input = input_test;

            feedforward();

            int n = output.size();
            //for printing purposes
            for (int i=0;i<n;i++){
                cout << output[i] << ' ';
            }
        }
    

};

void test_feedforward(){
    SingleExampleNeuralNetwork neural;

    //initialising input and weights
    neural.input = {1.0,2.0,3.0};
    neural.weights1 = {{1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}};
    neural.bias1 = {0,0,0,0,0};
    neural.weights2 = {{1.0,0.5,0.25,0.5,1.0}};
    neural.bias2 = {0,0,0,0,0};
    neural.output = {};

    neural.feedforward();

    for (auto i: neural.output)
    cout << i << ' ';
}

////TESTS////

void test_back_prop(){
    SingleExampleNeuralNetwork neural;

    //initialising input and weights
    neural.input = {1.0,2.0,3.0};
    neural.weights1 = {{1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}};
    neural.bias1 = {0,0,0,0,0};
    neural.weights2 = {{1.0,0.5,0.25,0.5,1.0}};
    neural.bias2 = {0,0,0,0,0};
    neural.output = {};
    neural.Y_vals = {0};

    neural.feedforward();

    cout << neural.output[0] << endl;;

    neural.back_prop();

    for (auto i: neural.bias1)
    cout << i << ' ';

}

void test1(int epochs){
    SingleExampleNeuralNetwork neural;

    //initialising input and weights
    neural.input = {1.0,2.0,3.0};
    neural.weights1 = {{1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}};
    neural.bias1 = {0,0,0,0,0};
    neural.weights2 = {{1.0,0.5,0.25,0.5,1.0}, {1.0,0.5,0.25,0.5,1.0}, {1.0,0.5,0.25,0.5,1.0}};
    neural.bias2 = {0,0,0};
    neural.output = {};
    neural.Y_vals = {0,1,0};

    neural.run(epochs);
}


//test for fully functioning 1 hidden layer neural network
void test2(int epochs){
    NeuralNetwork neural;

    neural.all_inputs = {{1.0,2.0,3.0},{1.0,1.0,0.5},{1.0,1.5,2.0}};
    neural.weights1 = {{1.0,0.5,0.25}, {-1.0,0.5,0.25}, {1.0,-0.5,0.25}, {1.0,-0.5,0.25}, {1.0,0.3,0.55}, {0.1,0.4,0.75}, {-1.0,-0.5,-0.25}, {-1.0,0.45,0.25}};
    neural.bias1 = {0.2,0.3,0.4,0.5,-0.1,0.2,-0.5,0.2};
    neural.weights2 = {{1.0,0.5,0.25,0.5,1.0,-0.3,0.1,0.1}, {1.0,-0.5,0.25,0.5,-1.0,-0.5,0.6,-0.1}};
    neural.bias2 = {0.1, 0.5};
    neural.all_outputs = {{},{},{}};
    neural.all_Y_vals = {{1,0},{0,0},{1,1}};

    neural.train(epochs);


    //this just prints out the output for comparison
    int n = neural.all_outputs.size();
    int m = neural.all_outputs[0].size();

    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            cout << neural.all_outputs[i][j] << ' ';
        }
        cout << "\n";
    }
    neural.run_single({-0.6,0.6,1.2});
}