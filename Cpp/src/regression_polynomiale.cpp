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
void in_main_runner(datasetp data_couple, Eigen::VectorXf Y_fit){

    /** Preparing data */
    float xmin, xmax, ymin, ymax; // Values to reshape plot size to fit in windows(HEIGHT, WIDTH)
    float padding_x = 0.1f ,padding_y = 0.1f; //percentage
    int n = (*(data_couple.pX)).size(); //nb of samples

    /** Reframing values in windows 
     * [xmin, xmax], [ymin, ymax] -->  [- WIDTH/2 , WIDTH/2], [- HEIGHT/2 , HEIGHT/2]
     * X = (x1, ..., xn) [a.u] (aribitrary unit)
     * x_min = min(X) & x_max = max(X) [a.u]
     * x_min < xi < x_max [a.u]
     * => 0 < (xi - xmin)/(xmax - xmin) < 1 [without unit] centered & normalized
     * => -WIDTH/2 < WIDTH*(xi - xmin)/(xmax - xmin) - WIDTH/2 < WIDTH/2 [pixel]
     * symetrical solution for y values...
     */
    xmin = (*(data_couple.pX)).minCoeff(); 
    xmax = (*(data_couple.pX)).maxCoeff();
    //note : ymin, ymax depend from Ŷ and Y
    ymin = Y_fit.minCoeff() < (*(data_couple.pY)).minCoeff() ? Y_fit.minCoeff() : (*(data_couple.pY)).minCoeff();
    ymax = Y_fit.maxCoeff() < (*(data_couple.pY)).maxCoeff() ? (*(data_couple.pY)).maxCoeff() : Y_fit.maxCoeff();

    // Padding making windows 10% zoom out
    padding_x = 10.0*(xmax-xmin)/100;
    padding_y = 10.0*(ymax-ymin)/100;
    xmin = xmin - padding_x;
    ymin = ymin - padding_y;
    xmax = xmax + padding_x;
    ymax = ymax + padding_y;

    //resolution of grid in arbitrary unit [a.u] : one line every ...
    vec2 grid_xy_resolution = {(xmax-xmin)/10.0, (ymax-ymin)/10.0};  

    /** PLOTTING */

    /**
     * Suppose we want a centered grid on l0 = (xmax+xmin)/2 [a.u.]
     * and we want the line i every res_x : li = l0 + i*resx [a.u.]
     * In plot window one want the li line in the center p(l0) = 0 [pixel]
     * We use reframe_values function such that :
     * res_x_px = p(li) - p(l_(i-1))= reframe_values(li) - reframe_values(l_(i-1)) [pixel]
     *          = (WIDTH  / (xmax - xmin)) * res_x [pixel]
     */

    float res_x_px = (WIDTH  / (xmax - xmin)) * grid_xy_resolution[0];
    float res_y_px = (HEIGHT / (ymax - ymin)) * grid_xy_resolution[1];
    set_grid(res_x_px, //convert res_x [a.u] --> [px]
             res_y_px); //idem pour res_y
    set_grid_text(res_x_px,  res_y_px, xmin,  xmax,  ymin,  ymax);


    // Blue point on (0, 0) [px] (of pixel reference frame)
    plot_couple({0.0f, 0.0f},  {0.0f, 0.0f, 1.0f}); 
    draw_text( 5.0, 5.0f, to_string((int) (xmax/2 + xmin/2)));


    data_couple.plot_dataset(xmin, xmax, ymin, ymax); // Data points reframed

    /**Fitting **/
    for(int i = 1; i < Y_fit.size() ; i++){
        plot_segment({reframe_values((*(data_couple.pX))(i-1), xmax, xmin, WIDTH),
                    reframe_values(Y_fit(i-1), ymax, ymin, HEIGHT)},
                    {reframe_values((*(data_couple.pX))(i), xmax, xmin, WIDTH),
                    reframe_values(Y_fit(i), ymax, ymin, HEIGHT)},
                    {0.0f, 0.0f, 1.0f});
        
    }
}



// --- main ---
Engine engine("Plot graph"); //Start the engine before main
int main () {
    
    /** SAMPLES **/
    int n = 400; // points number
    int d = 3; // degree of polynom fit
    float xmax = 430;
    float xmin = -430;

    Eigen::VectorXf X(n) ; 
    Eigen::VectorXf Y(n) ; // Y vector (before noising data)

    datasetp data_couple(&X, &Y);
    

    Eigen::VectorXf w_th(d+1);
    w_th << 0.000003f, 0.0f, -0.5f, 0.0f; //parameters values 

    /** NOISE : Gaussian **/
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 10.0f); //(mean ; std)  

    /** SYSTEM MATRIX **/
    int n_ = n; //matrix height : nb of points
    int m_ = d+1; //matrix width : nb of params
    Eigen::MatrixXf A  = Eigen::MatrixXf::Zero(n_, m_);
    Eigen::MatrixXf AT; //transpose
    Eigen::MatrixXf AT_A; // transpose * matrix (for system resolution)

    /** OPTIMAL FIT & PARAMETERS **/
    Eigen::VectorXf w(d+1);
    Eigen::VectorXf Y_fit = Eigen::VectorXf::Zero(n);

    

    /** Y set of points generation **/

    for (int i = 0; i < n; i++) {
        float x = i*(xmax-xmin)/n + xmin ;
        float y = 0.0f;
        for (int j = 0; j <= d ; j++) y+= pow(x,d-j)*w_th(j);
        X(i) = x;
        Y(i) = y;
    }

    // add noise
    for (int i = 0; i < n; i++) {
        Y(i) += distribution(generator);
    }


    // definition of system of equation : A.w = Ŷ
    for(int i = 0; i < n; i++){
        for(int j = 0; j <= d ; j++){
            A(i,j) = pow(X[i], d-j);
        }
        
    }

    AT = A.transpose();   
    AT_A = AT*A; 
    
    // resolution
    w =  (AT_A.inverse()) * AT  * Y;

    cout << "w_th = " << w_th.transpose() << endl; 
    cout << "w = " << w.transpose() << endl; 

    for (int j = 0; j <= d ; j++){
        Y_fit = Y_fit.array() + w(j)*( X.array().pow(d-j) );
    } 
    cout << "Rsquared = " << data_couple.R_squared(Y_fit) <<endl;

    /** PLOT REGRESSION **/
    engine.main_runner(in_main_runner, data_couple, Y_fit);

} 