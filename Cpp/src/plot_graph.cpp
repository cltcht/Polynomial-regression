#include <Eigen/Dense>
#include "../include/plot_graph.hpp"
#include <GL/glut.h>

/**
 * @brief Construct a new Engine:: Engine object
 * 
 * @param title Title of the window
 */
Engine::Engine(const char* title) {
    // --- Init GLUT ---
    int argc = 0;
    glutInit(&argc, nullptr);  // ← obligatoire avant glutBitmapCharacter
    // --- Init GLFW ---
    if (!glfwInit()) {
        cerr << "failed to init glfw";
        exit(EXIT_FAILURE);
    }

    // --- Create Window ---
    window = glfwCreateWindow(WIDTH, HEIGHT, title, nullptr, nullptr);
    if (!window) {
        cerr << "failed to create window";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);
}


/**
 * @brief Runs the engine (insert before main !!)
 * 
 */
void Engine::run() {
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // set origin to centre
    double halfWidth = WIDTH / 2.0f, halfHeight = HEIGHT / 2.0f;
    glOrtho(-halfWidth, halfWidth, -halfHeight, halfHeight, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


/**
 * @brief Sets a grid in background of the graph
 * 
 */
void set_grid(float res_x_px, float res_y_px){

    int nb_lignes_h = HEIGHT/res_x_px;
    int nb_lignes_v = WIDTH/res_y_px;

    glBegin(GL_LINES);
    glLineWidth(0.1f);
    glColor3f(0.5f, 0.5f, 0.5f);
    // Vertical lines
    int i = 1; //i=0 : see "(0, 0) origin plot"
    while(i * res_y_px < WIDTH/2){
        glVertex2f(i  * res_y_px, -HEIGHT/2); // start
        glVertex2f(i  * res_y_px, +HEIGHT/2);  // end
        glVertex2f(-i * res_y_px, -HEIGHT/2); // start
        glVertex2f(-i * res_y_px, +HEIGHT/2);  // end
        i++;

    };

    // Horizontal lines
    i = 1;
    while(i * res_x_px < HEIGHT/2){
        glVertex2f(-WIDTH/2.0f,  i*res_x_px); // start
        glVertex2f(WIDTH/2.0f,   i*res_x_px);  // end
        glVertex2f(-WIDTH/2.0f, -i*res_x_px); // start
        glVertex2f(WIDTH/2.0f,  -i*res_x_px);  // end 
        i++;

    };



    //Coordinate system in white, origin: (0, 0)
    glLineWidth(1.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    glVertex2f(0.0f, -HEIGHT/2.0f); // start
    glVertex2f(0.0f, HEIGHT/2.0f);  // end
    glVertex2f(-WIDTH/2.0f, 0.0f); // start
    glVertex2f(WIDTH/2.0f, 0.0f);  // end

    glEnd();
    
}


void set_grid_text(float res_px_h, float res_px_v, 
              float xmin, float xmax, float ymin, float ymax){
    // --- Texte---
    float res_au_x = (xmax - xmin) / WIDTH  * res_px_v;  // res en [a.u]
    float res_au_y = (ymax - ymin) / HEIGHT * res_px_h;

    float x0_au = (xmax + xmin) / 2.0f;  // centre en [a.u]
    float y0_au = (ymax + ymin) / 2.0f;

    glColor3f(1.0f, 1.0f, 1.0f);
    int i = 1;
    while (i * res_px_v < WIDTH/2) {
        draw_text( i * res_px_v, 5.0f, to_string((int)(x0_au + i * res_au_x)));
        draw_text(-i * res_px_v, 5.0f, to_string((int)(x0_au - i * res_au_x)));
        i++;
    }
    i = 1;
    while (i * res_px_h < HEIGHT/2) {
        draw_text(5.0f,  i * res_px_h, to_string((int)(y0_au + i * res_au_y)));
        draw_text(5.0f, -i * res_px_h, to_string((int)(y0_au - i * res_au_y)));
        i++;
    }
}

/**
 * @brief Plot a segment in graph
 * 
 * @param xy0 coordinates (x, y) of segment's start
 * @param xy1 coordinates (x, y) of segment's end
 * @param color segment color in graph
 */
void plot_segment(vec2 xy0, vec2 xy1, vec3 color){
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glColor3f(color[0], color[1], color[2]);
    glVertex2f(xy0[0], xy0[1]); // start
    glVertex2f(xy1[0], xy1[1]);  // end

    glEnd();
}

/**
 * @brief Plot a couple (x, y) in a graph
 * 
 * @param pos coordinates (x, y)
 * @param color point color in graph
 */
void plot_couple(vec2 pos, vec3 color){
    float segments = 5000;
    float r = 2;
    glColor3f(color[0], color[1], color[2]);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(pos[0], pos[1]);
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * M_PI * i/segments;
        float x = cos(angle) * r;
        float y = sin(angle) * r;  
        glVertex2f(x + pos[0], y + pos[1]);
    }
    glEnd();

}



//overload
Eigen::VectorXf reframe_values(Eigen::VectorXf v, float v_max, float v_min, float screen_dimension){
    /** Reframing values in windows
     * X = (x1, ..., xn) [a.u] (arbitrary unit)
     * x_min = min(X) & x_max = max(X) [a.u]
     * x_min < xi < x_max 
     * => 0 < (xi - xmin)/(xmax - xmin) < 1 [without unit] centered & normalized
     * => -WIDTH/2 < WIDTH*(xi - xmin)/(xmax - xmin) - WIDTH/2 < WIDTH/2 [pixel]
     * symetrical solution for y values...
     */
    return ((screen_dimension/(v_max-v_min))*(v- Eigen::VectorXf::Ones(v.size())*v_min)
             - (screen_dimension/2)*Eigen::VectorXf::Ones(v.size()));
}
float reframe_values(float f, float v_max, float v_min, float screen_dimension){
    /** Reframing values in windows
     * X = (x1, ..., xn) [a.u.]
     * x_min = min(X) & x_max = max(X) [a.u.]
     * x_min < xi < x_max 
     * => 0 < (xi - xmin)/(xmax - xmin) < 1 [without unit] centered & normalized
     * => -WIDTH/2 < WIDTH*(xi - xmin)/(xmax - xmin) - WIDTH/2 < WIDTH/2 [pixel]
     * symetrical solution for y values...
     */

    return ((screen_dimension/(v_max-v_min))*(f - v_min) - (screen_dimension/2));
}


void draw_text(float x, float y, const string& text) {
    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(x, y);
    for (char c : text) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
    }
}
