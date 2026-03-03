#ifndef DATA_SETP
#define DATA_SETP

#include <iostream>
#include <Eigen/Dense>

#include "../include/plot_graph.hpp"

using namespace glm;
using namespace std;


struct datasetp{
    Eigen::VectorXf  *pX ;
    Eigen::VectorXf  *pY ;

    //constructeur
    datasetp(Eigen::VectorXf *vec_X, Eigen::VectorXf *vec_Y);
    void print_data();
    void plot_dataset(float xmin, float xmax, float ymin, float ymax);
    void plot_dataset3D(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
    float R_squared(Eigen::VectorXf Y_fit);
};

#endif //DATA_SETP
