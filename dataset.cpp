#include <iostream>
#include <vector>
#include <fstream>


typedef unsigned char uchar;

uchar** read_mnist_images(std::string end_path, int& number_of_images, int& image_size) {

    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    //typedef unsigned char uchar;

    std::ifstream file("/Users/leopitsillides/Documents/GitHub/Neural_network/digit data/"+end_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + end_path + "`!");
    }
}

uchar* read_mnist_labels(std::string end_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file("/Users/leopitsillides/Documents/GitHub/Neural_network/digit data/"+end_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + end_path + "`!");
    }
}

std::vector< std::vector<float> > convert_images_to_array(uchar** dataset, int number_of_images, int image_size){
    std::vector< std::vector<float> > new_dataset;

    for (int i=0;i<number_of_images;i++){
        std::vector<float> each_image;
        for (int j=0;j<image_size;j++){
            each_image.push_back((float)dataset[i][j]);
        }
        new_dataset.push_back(each_image);
    }
    return new_dataset;
}

std::vector< std::vector<float> > convert_labels_to_array(uchar* dataset, int number_of_labels, int label_size){
    std::vector< std::vector<float> > new_dataset;

    for (int i=0;i<number_of_labels;i++){

        std::vector<float> each_image(label_size);

        for (int j=0;j<label_size;j++){
            if (j == (int)dataset[i]){
                each_image[j] = 1;
            }
            else {
                each_image[j] = 0;
            }
        }
        new_dataset.push_back(each_image);
    }
    return new_dataset;
}


int main(){ 
    int image_size;
    int label_size = 10;
    int train_size;
    int test_size;

    uchar** X_train_init = read_mnist_images("train-images.idx3-ubyte", train_size, image_size);
    uchar** X_test_init = read_mnist_images("t10k-images.idx3-ubyte", test_size, image_size);
    uchar* Y_train_init = read_mnist_labels("train-labels.idx1-ubyte", train_size);
    uchar* Y_test_init = read_mnist_labels("t10k-labels.idx1-ubyte", test_size);

    
    std::vector< std::vector<float> > X_train = convert_images_to_array(X_train_init, train_size, image_size);
    std::vector< std::vector<float> > X_test = convert_images_to_array(X_test_init, test_size, image_size);
    std::vector< std::vector<float> > Y_train = convert_labels_to_array(Y_train_init, train_size, label_size);
    std::vector< std::vector<float> > Y_test = convert_labels_to_array(Y_test_init, test_size, label_size);
    
    
    return 0;
}