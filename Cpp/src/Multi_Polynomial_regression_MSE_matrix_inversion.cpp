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
#include "../include/plot_graph_3D.hpp"


// --- main ---
Engine3D engine("Plot graph : scroll = zoom / left click+drag = rotate"); //Start the engine before main
int main () {
    
    int m; // X : points number
    int n; // Y : points number
    int d_x ; // degree of X-polynom fit
    int d_y ; // degree of Y-polynom fit
    float xmin; 
    float xmax;
    float ymin; 
    float ymax;

    Eigen::VectorXf w_th_x; 
    Eigen::VectorXf w_th_y;
    Eigen::VectorXf w_x; 
    Eigen::VectorXf w_y;
    Eigen::VectorXf w;

    float mean, std; //noise generator


    /** SAMPLES : Demo Z = fct(X^2; Y^2) **/
    bool demo_simple = false;
    if (demo_simple){
        m = 100; // X : points number
        n = 100; // Y : points number
        d_x =2 ; // degree of X-polynom fit
        d_y =2 ; // degree of Y-polynom fit
        xmin = -5000.0; 
        xmax = 5000;
        ymin = -3000; 
        ymax = 3000;

        w_th_x.resize(d_x+1); 
        w_th_y.resize(d_y+1);
        w_x.resize(d_x+1); 
        w_y.resize(d_y);
        w.resize(d_x+d_y+1);

        w_th_x << -0.0002f, 0.0f, 0.0f; //parameters values 
        w_th_y << 0.0005f, 0.2f, 0.0f; //parameters values
        //Gaussian(mean ; std) noise
        mean = 0.0f, 
        std = 100.0f; 
    } 

    
    /** SAMPLES : Demo Gaussian **/
    // If T = exp(X^2+ Y^2), we solve Z = ln(T) = X^2+ Y^2
    // and plot T = exp(Z)
    // For today, we generate dataset = fct(X² , Y²) and assume it's a sqrt(gaussian) data.
    // But in real usage, one can suppose we got raw data from csv datafile from experiment results
    // Then we could apply our algorithm and use our 3D engine to plot fit !
    bool demo_gaussian = true;
    if (demo_gaussian){
        if (demo_simple) {
            cerr << "Both demos are ON ! Please choose : demo_gaussian XOR demo_simple !" << endl;
            return 1;
        }
        m = 100; // X : points number
        n = 100; // Y : points number
        d_x =2 ; // degree of X-polynom fit
        d_y =2 ; // degree of Y-polynom fit
        xmin = -5000.0f; xmax = 5000.0f;
        ymin = -5000.0f; ymax = 5000.0f;

        w_th_x.resize(d_x+1); 
        w_th_y.resize(d_y+1);
        w_x.resize(d_x+1); 
        w_y.resize(d_y);
        w.resize(d_x+d_y+1);

        w_th_x <<  -2.5e-7f, 0.0f, 4.5f;  //parameters values 
        w_th_y <<  -2.5e-7f, 0.0f, 3.45f;  //parameters values 
        //Gaussian(mean ; std) noise
        mean = 0.0f, 
        std = 0.1f; 
    }

    Eigen::VectorXf X(m) ; 
    Eigen::VectorXf Y(n) ;
    Eigen::MatrixXf Z(n, m);
    Eigen::MatrixXf Z_fit(n, m);
    Eigen::VectorXf Z_fit_vec(n*m) ; 
    Eigen::VectorXf Z_vec(n*m) ; 

    string equation = "Z = ";

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, std); //(mean ; std)  


    /** SYSTEM MATRIX **/
    int n_ = n*m; //matrix height : nb of points
    int m_ = d_x+d_y+1; //matrix width : nb of params
    Eigen::MatrixXf A  = Eigen::MatrixXf::Zero(n_, m_);
    Eigen::MatrixXf AT; //transpose
    Eigen::MatrixXf AT_A; // transpose * matrix (for system resolution)

    /** OPTIMAL FIT & SYSTEM DEFINITION **/

    /** X, Y set of points generation **/
    for (int j = 0; j < m; j++) X(j) = j*(xmax-xmin)/m + xmin ;
    for (int i = 0; i < n; i++) Y(i) = i*(ymax-ymin)/n + ymin ;

    /** definition of Z points to fit later */

    Z = Eigen::MatrixXf::Zero(n, m);
    for(int i = 0; i < n ; i++){
        for (int j = 0 ; j < m ; j++){
            for(int k = 0; k < w_th_x.size(); k++){ 
                Z(i, j)+= pow(X(j),d_x-k)*w_th_x(k);
                if ((i == 0 && j == 0)) equation += to_string(w_th_x(k))+"*x^"+to_string(d_x-k)+"+ ";
            }
            for(int k = 0; k < w_th_y.size(); k++){
                Z(i, j)+= pow(Y(i),d_y-k)*w_th_y(k) + distribution(generator) ;
                if ((i == 0 && j == 0)) equation += to_string(w_th_y(k))+"*y^"+to_string(d_y-k)+"+ ";
            }
            Z_vec(j*n+i) = Z(i,j); // Z_vec = (Z00, ..., Z0m, Z10, ...Z1m ... Zn0... Znm)
        }
    }

    //cout << "A size: " << n_ << "x" << m_ << endl;

    // Definition of system of equation : A.w = Ẑ_vec
    for(int j = 0; j < m; j++){ // j = 0 1 2
        for(int i = 0; i < n ; i++){ // i = 0 1 2
            for(int i_x = 0; i_x < d_x ; i_x++) A(i+n*j, i_x) = pow(X(j), d_x-i_x);
            for(int i_y = 0; i_y < d_y ; i_y++) A(i+n*j, d_x+i_y) = pow(Y(i), d_y-i_y);
            A(i+n*j, d_x+d_y) = 1;
        }
    }
    
    AT = A.transpose();   
    AT_A = AT*A; 
    
    // resolution
    w =  (AT_A.inverse()) * AT  * Z_vec;

    w_x << w.segment(0, d_x), w(w.size() - 1);  // coeffs X + constante
    w_y = w.segment(d_x, d_y);// coeffs Y

    //cout << "w_th = " << w_th_x.transpose() << " " <<w_th_y.transpose() << endl; 
    //cout << "w_x = " << w_x.transpose() << endl; 
    //cout << "w_y = " << w_y.transpose() << endl; 

    Z_fit_vec =  A * w;
    Z_fit = Eigen::Map<Eigen::MatrixXf>(Z_fit_vec.data(), n, m);
    if (demo_gaussian){
        Z = Z.array().exp();
        Z_fit = Z_fit.array().exp();
    }

    //cout << equation << endl;
    /** PLOT REGRESSION **/

    engine.set_camera(X.minCoeff(), X.maxCoeff(), Y.minCoeff(), Y.maxCoeff(), 
                        Z_fit.minCoeff() < Z.minCoeff() ? Z_fit.minCoeff() : Z.minCoeff(), 
                        Z_fit.maxCoeff() < Z.maxCoeff() ? Z.maxCoeff() : Z_fit.maxCoeff(), 
                        45.0f, 0.8f);

    engine.main_runner(X, Y, Z, Z_fit);

} 