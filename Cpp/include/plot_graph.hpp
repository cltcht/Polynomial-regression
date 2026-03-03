#ifndef PLOT_GRAPH  
#define PLOT_GRAPH

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>  
#include <iostream> 

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
 * @brief Graphic engine
 * 
 */
struct Engine {
    
    GLFWwindow* window;

    // prototypes
    Engine(const char* title); 
    void run();                 

    template <typename T1, typename T2>
    void main_runner(void (*func)(T1, T2), T1 data1, T2 data2) {
        
        while (!glfwWindowShouldClose(window)) {
            run();
            func(data1, data2);
            glfwSwapBuffers(window);
            glfwWaitEvents();
        }
    }
};


void plot_segment(vec2 xy0, vec2 xy1, vec3 color);
void plot_couple(vec2 pos, vec3 color);
void set_grid(float res_h, float res_v);
void set_grid_text(float res_px_h, float res_px_v, 
              float xmin, float xmax, float ymin, float ymax);


//overload
Eigen::VectorXf reframe_values(Eigen::VectorXf v, float v_max, float v_min, float screen_dimension);
float reframe_values(float v, float v_max, float v_min, float screen_dimension);

void draw_text(float x, float y, const string& text);
#endif //PLOT_GRAPH