#ifndef PLOT_GRAPH_3D  
#define PLOT_GRAPH_3D

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>  
#include <iostream> 
#include "../include/data_setp.hpp"

using namespace glm;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif //M_PI

#ifndef WIDTH
#define WIDTH  800
#endif //WIDTH

#ifndef HEIGHT
#define HEIGHT  600
#endif //HEIGHT

using namespace glm;
using namespace std;

// PROTOYPES

/**
 * @brief 3D Graphic engine
 * 
 */
struct Engine3D {
    
    GLFWwindow* window;

    // Camera parameters
    float xmin, xmax;
    float ymin, ymax;
    float zmin, zmax;
    float fovy_deg;
    float k;

    //user interface for zoom and rotation
    float yaw, pitch;
    double lastMouseX, lastMouseY;
    bool mousePressed;

    // prototypes
    Engine3D(const char* title); 
    
    void set_camera(float xmin, float xmax,
                        float ymin, float ymax,
                        float zmin, float zmax,
                        float fovy_deg, float k);

    void run();
    void set_grid_3D(float res_x, float res_y, float res_z);
    void plot_couple_3D(vec3 pos, vec3 color);
    void plot_data_3D(vec3 pos, vec3 color); 
    void plot_dataset3D(datasetp data);
    void draw_surface_wireframe(const Eigen::VectorXf& X, 
                                const Eigen::VectorXf& Y, 
                                const Eigen::MatrixXf& Z, 
                                const vec3& color);
              

    void main_runner(Eigen::VectorXf X, Eigen::VectorXf Y, Eigen::MatrixXf Z, Eigen::MatrixXf Z_fit) {
        while (!glfwWindowShouldClose(window)) {
            run();
            //Grid with res.{x, y, z} = 10% of plot amplitude
            set_grid_3D(0.3f*(this->xmax-this->xmin)/2, 
                        0.3f*(this->ymax-this->ymin)/2, 
                        0.2f*(this->zmax-this->zmin)/2);
            draw_surface_wireframe(X, Y, Z_fit, {0.0f, 0.6f, 0.6f});
            for(int j = 0; j < X.size(); j++){
                for(int i = 0; i < Y.size(); i++){
                    plot_couple_3D({ X(j), Y(i), Z(i, j) }, {1.0f, 0.0f, 0.0f});
                    // plot_couple_3D({ X(j), Y(i), Z_fit(i, j) }, {0.0f, 1.0f, 1.0f});
                }
            }

            glfwSwapBuffers(window);
            glfwWaitEvents();  // static plot
        }
    }
};

void draw_text_3D(float x, float y, float z, const string& text);


#endif //PLOT_GRAPH