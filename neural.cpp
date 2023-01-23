#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

float sigmoid(int z);
float mse(vector<int> y_true, vector<float> y_check);

int main(){

    vector<int> y = {1,0,0,1,1,0,1};
    vector<float> y_hat = {0.24,0.1,0.3,0.91,0.87,0.4,0.665};

    cout << mse(y, y_hat) << endl;
    return 0;
}


float sigmoid(int z){
    return 1/(1+exp(-z));
}

float mse(vector<int> y_true, vector<float> y_check){
    float error;
    int n = sizeof(y_true)/sizeof(y_true[0]);

    for (int i = 0; i < n; i++){
        error += pow(y_check[i] - y_true[i], 2);
    }
    return error/n;
}