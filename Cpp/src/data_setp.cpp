#include <iostream>
#include <Eigen/Dense>

#include "../include/data_setp.hpp"
#include "../include/plot_graph.hpp"
#include "../include/plot_graph_3D.hpp"




/**
 * @brief Construct a new datasetp::datasetp object
 * 
 * @param *vec_X Eigen::VectorXf pointer 
 * @param *vec_Y Eigen::VectorXf pointer
 */
datasetp::datasetp(Eigen::VectorXf *vec_X, Eigen::VectorXf *vec_Y) : pX(vec_X), pY(vec_Y){
    if ((*vec_X).size() != (*vec_Y).size())
    {
        cerr << "Vectors don't have the same size";
        exit(EXIT_FAILURE);
    }
    pX = vec_X;
    pY = vec_Y;
}

                               
/**
 * @brief print dataset data in table
 * 
 * @return * void 
 */
void datasetp::print_data(){
        //printf("i | x ; y\n");
        // for(int i = 0; i < (*pX).size(); i++){
        //     cout << i << " | " << (*pX)(i) <<  " | " << (*pY)(i) << endl;
        // }  
    }

/**
 * @brief plot datasset in graph
 * 
 */
void datasetp::plot_dataset(float xmin, float xmax, float ymin, float ymax){

    for (int i=0; i<(*pX).size(); i++){     
        plot_couple({ reframe_values((*pX)(i), xmax, xmin, WIDTH),
                      reframe_values((*pY)(i), ymax, ymin, HEIGHT)}, {1.0f, 0.0f, 0.0f});


    }
}

/**
 * @brief Compute Rsquared from a given dataset and a fit.
 * 
 * @param Y_fit 
 */
float datasetp::R_squared(Eigen::VectorXf Y_fit){
    if ((*pY).size() != Y_fit.size()) {
            cerr << "R_squared : dataset.Y.size() != Y_fit.size()" << endl;
            exit(EXIT_FAILURE);
    }
    float mean = (*pY).mean();
    float SS_mean = ((*pY) - Eigen::VectorXf::Ones((*pY).size())*mean).squaredNorm();
    float SS_fit = ((*pY) - Y_fit).squaredNorm();

    return (SS_mean - SS_fit)/SS_mean ;
}



