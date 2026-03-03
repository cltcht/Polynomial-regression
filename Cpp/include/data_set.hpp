#ifndef DATA_SET
#define DATA_SET

#include <iostream>
#include <Eigen/Dense>

#include "../include/plot_graph.hpp"

using namespace glm;
using namespace std;

/**
 * @brief struct that combines two vector
 * 
 */
struct dataset{
    Eigen::VectorXf  X ;
    Eigen::VectorXf  Y ;


    dataset(Eigen::VectorXf& vec_X, Eigen::VectorXf& vec_Y);
    void print_data();
    void plot_dataset();
    float R_squared(Eigen::VectorXf Y_fit);

};


#endif //DATA_SET
