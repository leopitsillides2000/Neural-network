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

class NeuralNetwork{
    public:

        //attributes for the NeuralNetwork class
        vector<float> input;
        vector<vector<float>> weights1; //dimension (size of hidden layer, size of input)
        vector<vector<float>> weights2; //dimension (size of output, size of hidden layer)
        vector<float> output;
        vector<float> Y_vals;
        float lr = 0.5; //learning rate could be an attribute to class object

        //activation function: sigmoid
        static float sigmoid(float x){

            float val = 1/(1+exp(-x));

            return val;
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
        static vector<float> apply_weights(vector<float> layer, vector<vector<float>> weights){
            vector<float> new_layer;

            int n = weights.size();

            cout << n << endl; //debug

            cout << "before" << endl; //debug

            //for each node in new_layer calculate the dot product of the orignal layer with the weights
            for (int i=0; i<n; i++){

                //new_layer[i] = sigmoid(dot_product(layer, weights[i]));
                new_layer.insert(new_layer.end(), sigmoid(dot_product(layer, weights[i]))); ///////Vectors are not indexable in c++

                //debug segmentation fault
                cout << i << ": " << new_layer[i] << endl;
            }
            cout << "after" << endl; //debug
            return new_layer;
        }


        void feedforward(){
            //calculates first layer
            vector<float> layer1 = apply_weights(input, weights1);
            //calculates output layer
            output = apply_weights(layer1, weights2);
        }

        static float sigmoid_der(float x){
            return sigmoid(x)*(1-sigmoid(x));
        }

        float mse(vector<float> y_true, vector<float> y_check){
            float error;
            int n = y_true.size();

            for (int i = 0; i < n; i++){
                error += pow(y_check[i] - y_true[i], 2);
            }
            return error/n;
        }

        float error(float y_true, float y_out){
            return 1/2*pow((y_true - y_out),2);
        }

        void grad_descent(float weight, float z, float a){
            weight = weight - a*sigmoid_der(z);
        }

        //Little cofused!! Do you need to feed forward for each training example first then do the back prpagation
        //Or do you apply the backpropogation after the feedforard of each training example?
        static void update_weights(vector<float> current_layer, vector<float> true_layer, vector<vector<float>> current_weights){
            //for each output node
                //calculate error for each output node
                //adjusts weights accordingly using grad descent

            int n = current_layer.size();

            for (i=0;i<n;i++){
                //calculate error for each output node
                float cost = error(true_layer[i], current_layer[i]);

                int m = current_weights[i].size();

                for (j=0;j<m;j++){
                    //applies change of weights in-place
                    grad_descent(current_weights[i][j], cost, lr); //this is a little inefficent as the sigmoid_der is calculated eachh time despite z/cost being unchanged
                }
            }
            
        }

        void backprop(){
            

        }
    };

void test_feedforward(){
    NeuralNetwork neural;

    //initialising input and weights
    neural.input = {1.0,2.0,3.0};
    neural.weights1 = {{1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}, {1.0,0.5,0.25}};
    neural.weights2 = {{1.0,0.5,0.25,0.5,1.0}};
    neural.output = {};

    neural.feedforward();

    for (auto i: neural.output)
    cout << i << ' ';
}