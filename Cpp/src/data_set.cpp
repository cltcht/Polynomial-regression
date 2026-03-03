#include <iostream>
#include <Eigen/Dense>

#include "../include/data_set.hpp"
#include "../include/data_set.hpp"



/**
 * @brief Construct a new dataset::dataset object
 * 
 * @param vec_X 
 * @param vec_Y 
 */
dataset::dataset(Eigen::VectorXf& vec_X, Eigen::VectorXf& vec_Y) : X(vec_X), Y(vec_Y) 
    {
    if (vec_X.size() != vec_Y.size())
    {
        cerr << "vectors don't have the same size";
        exit(EXIT_FAILURE);
    }
    X = vec_X;
    Y = vec_Y;
    }


                               
/**
 * @brief print dataset data in table
 * 
 * @return * void 
 */
void dataset::print_data(){
    printf("i | x ; y\n");
    for(int i = 0; i < X.size(); i++){
        printf("%d | %.1f ; %.1f\n", i, X[i], Y[i]);
    }  
}




/**
 * @brief plot datasset in graph
 * 
 */
void dataset::plot_dataset(){
    for (int i=0; i<X.size(); i++){     
        plot_couple({X(i),Y(i)}, {1.0f, 0.0f, 0.0f});
    }
}

/**
 * @brief Compute Rsquared from a given dataset and a fit.
 * 
 * @param Y_fit 
 */
float dataset::R_squared(Eigen::VectorXf Y_fit){
    if (Y.size() != Y_fit.size()) {
            cerr << "R_squared : dataset.Y.size() != Y_fit.size()" << endl;
            exit(EXIT_FAILURE);
    }
    float mean = Y.mean();
    float SS_mean = (Y - Eigen::VectorXf::Ones(Y.size())*mean).squaredNorm();
    float SS_fit = (Y - Y_fit).squaredNorm();

    return (SS_mean - SS_fit)/SS_mean ;
}

