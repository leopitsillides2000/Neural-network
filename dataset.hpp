#ifndef DATASET_HPP // include guard
#define DATASET_HPP

#include "dataset.cpp"

extern std::vector< std::vector<long double> > X_train;
extern std::vector< std::vector<long double> > X_test;
extern std::vector< std::vector<long double> > Y_train;
extern std::vector< std::vector<long double> > Y_test;
extern void create_dataset();

#endif