#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class NeuralNetwork;

int main(){
    NeuralNetwork neural;

    //initialising 
    neural.input = ;
    neural.weights1 = ;
    neural.weights2 = ;
    neural.output

    return 0;
}

class NeuralNetwork{
    public:
        vector<int> input;
        vector<float> weights1;
        vector<float> weights2;
        vector<float> output;


        void sigmoid(vector<float> x){
            int n = sizeof(x)/sizeof(x[0]);

            for (i=0;i<n;i++){
                x[i] = 1/(1+exp(-x[i]));
            }

            };

        vector<float> sigmoid_der(vector<float> x){
            return sigmoid(x)*(1-sigmoid(x));
        };

        float mse(vector<float> y_true, vector<float> y_check){
            float error;
            int n = sizeof(y_true)/sizeof(y_true[0]);

            for (int i = 0; i < n; i++){
                error += pow(y_check[i] - y_true[i], 2);
            }
            return error/n;
        };

        float dot_product(vector<float> x_vec, vector<float> y_vec){
            int n = sizeof(y_true)/sizeof(y_true[0]);
            float total = 0;

            for (int i = 0; i<n; i++){
                total += x_vec[i]*y_vec[i];
                }
            return total;
        };

        void grad_descent(float weight, float z, float a){
            weight = weight - a*sigmoid_der(z);
        };

        void feedforward(vector<int> input, vector<float> weights1, vector<float> weights2){
            
            //input should be a vector of vectors
            //each vector in input is dot product with a vector of weights so weights1 is a vector of vectors
            //this then gives a simple vector so layer1 will be a vector
            //layer1 is dot product with weights2, a vector, to give a single output value

            vector<float> layer1 = sigmoid(dot_product(input, weights1));
            output = sigmoid(dot_product(layer1, weights2));
        };

        void backprop(){

        };
}