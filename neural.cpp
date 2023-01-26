#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class NeuralNetwork;

void test_feedforward();

int main(){
    //NeuralNetwork neural;
    //initialising input and weights
    //neural.input = ;
    //neural.weights1 = ;
    //neural.weights2 = ;
    //neural.output;

    test_feedforward();

    return 0;
}

class SingleExampleNeuralNetwork{
    public:

        //attributes for the NeuralNetwork class
        vector<float> input; 
        vector<vector<float>> weights1; //dimension (size of hidden layer, size of input)
        vector<float> bias1; // size of hidden layer
        vector<vector<float>> weights2; //dimension (size of output, size of hidden layer)
        vector<float> bias2; // size of output
        vector<float> layer1;
        vector<float> output; 
        vector<float> Y_vals;
        float lr = 0.5; //learning rate could be an attribute to class object

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
        static vector<float> apply_weights(vector<float> layer, vector<vector<float>> weights, vector<float> bias){
            vector<float> new_layer;

            int n = weights.size(); //this might be weights[0] e.g. vector that is length of next layer

            //for each node in new_layer calculate the dot product of the orignal layer with the weights
            for (int i=0; i<n; i++){

                //new_layer[i] = sigmoid(dot_product(layer, weights[i]));
                new_layer.insert(new_layer.end(), sigmoid(dot_product(layer, weights[i]) + bias[i])); ///////Vectors are not indexable in c++

                //debug segmentation fault
                cout << i << ": " << new_layer[i] << endl;
            }
            cout << "after" << endl; //debug
            return new_layer;
        }


        void feedforward(){

            //calculates hidden layer
            layer1 = apply_weights(input, weights1, bias1);
            //calculates output layer
            output = apply_weights(layer1, weights2, bias2);
            
        }

        static float sigmoid_der(float z){
            return sigmoid(z)*(1-sigmoid(z));
        }

        vector<float> update(vector<float> current_layer, vector<float> prev_layer, vector<vector<float>> current_weights, vector<float> current_bias, vector<float> Y_true){
            //for each node(j)
            int n = current_layer.size();
            int m = current_weights[0].size();
            vector<float> new_prev_layer; //need to calc prev layer with new weights and bias to apply update to each layer


            for (int j=0;j<n;j++){
                
                //term derived from chain rule
                //for gradient descent all updated parts(bias/weights/neurons) share this term
                float descent = 2*lr*sigmoid_der(inv_sigmoid(current_layer[j]))*(current_layer[j]-Y_true[j]); //maybe have to abs last part

                //update bias
                current_bias[j] = current_bias[j] - descent;

                for (int k=0;k<m;k++){
                    //update weight
                    float new_weight = current_weights[j][k] - prev_layer[k]*descent;
                    current_weights[j][k] = new_weight;

                    //keep track of updated nodes
                    new_prev_layer[k] = prev_layer[k] - new_weight*descent;
                }
            }


        return new_prev_layer; //may need to return new_prev_layer to use it again
        }

        void back_prop(){
            vector<float> pre_layer = update(output, layer1, weights2, bias2, Y_vals);
            vector<float> pre_pre_layer = update(layer1, input, weights1, bias1, pre_layer);
        }
    };

void test_feedforward(){
    SingleExampleNeuralNetwork neural;

    //initialising input and weights
    neural.input = {1.0,2.0,3.0};
    neural.weights1 = {{1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}};
    neural.weights2 = {{1.0,0.5,0.25,0.5,1.0}};
    neural.output = {};

    neural.feedforward();

    for (auto i: neural.output)
    cout << i << ' ';
}