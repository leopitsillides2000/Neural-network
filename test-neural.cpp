#include "dataset.hpp"
#include "neural.hpp"

void run1(int epochs, std::vector<int> hidden_layers, long double lr);

void run_mnist(int epochs, std::vector<int> hidden_layers, long double lr, std::vector< std::vector<long double> > X_train, std::vector< std::vector<long double> > Y_train, std::vector< std::vector<long double> > X_test, std::vector< std::vector<long double> > Y_test);

int main(){

    run1(5000, {32}, 0.5);

    //create_dataset();
    //std::cout << X_train[0][0] << std::endl;

    //run_mnist(3, {12}, 0.5, X_train, Y_train, X_test, Y_test);


    return 0;
}

//smaller testing
void run1(int epochs, std::vector<int> hidden_layers, long double lr){
    NeuralNetwork neural;

    //attributes to initialise
    neural.lr = lr; //learning rate
    neural.hidden_layers = hidden_layers; //hidden layers[i] = no of nodes in hidden layer i
    neural.all_inputs = {{1.3,-0.6,0.5}, {0.3,1.1,-0.2}, {1.0,1.0,0.4}, {0.9,0.1,-0.3}}; //4 training examples with 3 inputs each
    neural.all_Y_vals = {{0,1},{1,0},{0,1},{1,0}};

    neural.train(epochs);

    neural.run_single({1.33,-0.55,0.4});

}

//testing on mnist handwritten digit dataset
void run_mnist(int epochs, std::vector<int> hidden_layers, long double lr, std::vector< std::vector<long double> > X_train, std::vector< std::vector<long double> > Y_train, std::vector< std::vector<long double> > X_test, std::vector< std::vector<long double> > Y_test){
    NeuralNetwork neural;

    //vector< vector<long double> > all_inputs;
    //vector< vector<long double> > all_Y_vals;

    //attributes to initialise
    neural.lr = lr; //learning rate
    neural.hidden_layers = hidden_layers; //hidden layers[i] = no of nodes in hidden layer i

    neural.all_inputs = X_train;
    neural.all_Y_vals = Y_train;

    neural.train(epochs);

    std::cout << "EXAMPLE 1" << std::endl;
    std::cout << "weights: ";
    for (int i=0;i<10;i++){
        std::cout << neural.weights[1][7][i] << ' ';
    }
    std::cout << "\nbias: ";

    std::cout << neural.biases[1][7] << std::endl;

    //just a small test
    std::cout << "output: ";
    neural.run_single(X_test[0]);
    
    std::cout << "true: ";
    for (int i=0;i<10;i++){
        std::cout << Y_test[0][i] << ' ';
    }
    std::cout << '\n';

    std::cout << "EXAMPLE 2" << std::endl;
    std::cout << "\nweights: ";

    for (int i=0;i<10;i++){
        std::cout << neural.weights[1][2][i] << ' ';
    }
    std::cout << "\nbias: ";

    std::cout << neural.biases[1][2] << std::endl;

    std::cout << "output: ";
    neural.run_single(X_test[1]);

    std::cout << "true: ";
    for (int i=0;i<10;i++){
        std::cout << Y_test[1][i] << ' ';
    }
    
}