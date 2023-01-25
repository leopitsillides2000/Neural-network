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

        //activation function: sigmoid
        static float sigmoid(float x){

            float val = 1/(1+exp(-x));

            return val;
        }

        //calculates dot product for application iun feedforward
        static float dot_product(vector<float> x_vec, vector<float> y_vec){

            int n = sizeof(x_vec)/sizeof(x_vec[0]); //sizeof gives number of bytes so must divide by the number of bytes of the data type in the vector
            float total = 0;

            for (int i = 0; i<n; i++){
                total += x_vec[i]*y_vec[i];
            }

            return total;
        }

        //outputs a new layer given the original layer and the weights
        static vector<float> apply_weights(vector<float> layer, vector<vector<float>> weights){
            vector<float> new_layer;

            int n = sizeof(weights)/sizeof(weights[0]); /////////////This is giving the wrong output

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
            int n = sizeof(y_true)/sizeof(y_true[0]);

            for (int i = 0; i < n; i++){
                error += pow(y_check[i] - y_true[i], 2);
            }
            return error/n;
        }

        //void grad_descent(float weight, float z, float a){
        //    weight = weight - a*sigmoid_der(z);
        //}

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