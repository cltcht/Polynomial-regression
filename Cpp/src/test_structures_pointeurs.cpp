#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;

#include "../include/plot_graph.hpp"
#include "../include/data_set.hpp"
#include "../include/data_setp.hpp"

/**
 * @brief function to execute in main_runner
 * 
 * @param data_couple dataset to plot in graph
 */
void in_main_runner(dataset *data_couple, vec2 xy0, vec2 xy1){

    plot_couple({0.0f, 0.0f},  {0.0f, 0.0f, 1.0f});

    data_couple->plot_dataset();
    plot_segment(xy0, xy1, {0.0, 0.0, 1.0});
}


// --- main ---
Engine engine("Plot graph"); //Start the engine before main
int main () {
    
    // Definition vectors :
    int n = 3; // points number
    float xmax = WIDTH/2, ymax = HEIGHT/2;
    float xmin = -xmax, ymin = -ymax;

    Eigen::VectorXf X(n) ; 
    Eigen::VectorXf Y(n) ;

    Eigen::VectorXf w_th(2);
    w_th <<  0.5f, -8.0f; //theorical parameters values (before noising data)


    // for (int i = 0; i < n; i++) {
    //     float x = i*(xmax-xmin)/n + xmin ;
    //     float y = w_th(0)*x + w_th(1);
    //     // cout << x << " , "  << y << endl;
    //     X(i) = x;
    //     Y(i) = y;
    // }

    // // noise : gaussian noise mean = 0 and std = 10
    // std::default_random_engine generator;
    // std::normal_distribution<float> distribution(0.0f, 40.0f); 

    // for (int i = 0; i < n; i++) {
    //     Y(i) += distribution(generator);
    // }

    cout << "### 1 - Test modifier les variables de structure ###" << endl;
    cout << "X, Y <-- (1), (1)" <<endl;
    X = Eigen::VectorXf::Ones(n);
    Y = Eigen::VectorXf::Ones(n);

    dataset data_couple(X, Y);

    cout << "dataset : " << endl;
    data_couple.print_data();

    cout << "data_couple.X/Y <-- (0), (0)" << endl;
    data_couple.X = Eigen::VectorXf::Zero(n);
    data_couple.Y = Eigen::VectorXf::Zero(n);

    cout << "dataset : " << endl;
    data_couple.print_data();    

    cout << "\n### 2 - Test modifier les variables puis print la structure ###" << endl;
    cout << "X, Y <-- (1), (1)" <<endl;
    Eigen::VectorXf X2(n), Y2(n);
    
    X2 = Eigen::VectorXf::Ones(n);
    Y2 = Eigen::VectorXf::Ones(n);

    dataset data_couple_2(X2, Y2);    

    cout << "dataset : " << endl;
    data_couple_2.print_data();

    cout << "X, Y <-- (0), (0)" << endl;
    X2 = Eigen::VectorXf::Zero(n);
    Y2 = Eigen::VectorXf::Zero(n);

    cout << "dataset : " << endl;
    data_couple.print_data();    

    cout << "\n### 3 - Test modifier les variables de la structure avec pointeurs ###" << endl;
    
    Eigen::VectorXf X3(n), Y3(n);
    Eigen::VectorXf *pX, *pY;

    pX = &X3;
    pY = &Y3;

    cout << "*X, *Y <-- (1), (1)" <<endl;
    *pX = Eigen::VectorXf::Ones(n);
    *pY = Eigen::VectorXf::Ones(n);
    

    datasetp data_couple_3(pX, pY);

    data_couple_3.print_data();

    cout << "*X, *Y <-- (0), (0)" <<endl;
    *pX = Eigen::VectorXf::Zero(n);
    *pY = Eigen::VectorXf::Zero(n);

    data_couple_3.print_data();






    // int n_ = X.size();
    // int m_ = 2;
    // Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n_, m_);
    
    // for(int i = 0; i < n; i++){
    //     A(i,0) = X[i];
    //     A(i,1) = 1.0f;
    // }

    // Eigen::MatrixXf AT = A.transpose();   

    // Eigen::MatrixXf AT_A = AT*A; 
    

    // Eigen::VectorXf w =  (AT_A.inverse()) * AT  * Y;

    // cout << "w_th = ( a = " << w_th(0) << " ; b = " << w_th(1) << " ) " << endl; 
    // cout << "w = ( " << w(0) << " ; " << w(1) << " ) " << endl; 

    // Eigen::VectorXf Y_fit = w(0)*data_couple.X + w(1)*Eigen::VectorXf::Ones(X.size());


    // cout << "Rsquared = " << data_couple.R_squared(Y_fit) <<endl;

    // // --- PLOT REGRESSION --
    // engine.main_runner(in_main_runner, &data_couple, {X(0), Y_fit(0)}, {X(X.size()-1), Y_fit(Y_fit.size()-1)});

} 