#include <Eigen/Dense>
#include "../include/plot_graph_3D.hpp"
#include <GL/glut.h>

/**
 * @brief Construct a new Engine:: 3D Engine object
 * 
 * @param title Title of the window
 */
Engine3D::Engine3D(const char* title) {

    this->yaw = 0.0f;
    this->pitch = 0.0f;
    this->lastMouseX = 0.0;
    this->lastMouseY = 0.0;
    this->mousePressed = false;

    // --- Init GLUT ---
    int argc = 0;
    glutInit(&argc, nullptr);  //for glutBitmapCharacter
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

    glfwSetWindowUserPointer(window, this); // callBack on mouse button for zoom and left click rotation

    //ZOOM feature
    glfwSetScrollCallback(window, [](GLFWwindow* win, double xoffset, double yoffset) {
    auto* engine = static_cast<Engine3D*>(glfwGetWindowUserPointer(win));
    // yoffset > 0 : mouse (zoom in), yoffset < 0 : zoom out
    engine->k *= (1.0f - 0.1f * static_cast<float>(yoffset)); // 10% zoom
    // Zoom limits
    if (engine->k < 0.1f) engine->k = 0.1f;
    if (engine->k > 5.0f) engine->k = 5.0f;
    });

    //ROTATION feature
    glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
        auto* engine = static_cast<Engine3D*>(glfwGetWindowUserPointer(win));
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                engine->mousePressed = true;
                glfwGetCursorPos(win, &engine->lastMouseX, &engine->lastMouseY);
            } else if (action == GLFW_RELEASE) {
                engine->mousePressed = false;
            }
        }
    });
    //ROTATION feature
    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double xpos, double ypos) {
        auto* engine = static_cast<Engine3D*>(glfwGetWindowUserPointer(win));
        if (engine->mousePressed) {
            float dx = static_cast<float>(xpos - engine->lastMouseX);
            float dy = static_cast<float>(ypos - engine->lastMouseY);
            float sensitivity = 0.005f;
            engine->yaw   -= dx * sensitivity;
            engine->pitch += dy * sensitivity;
            // Upper/lower limits for rotation  
            if (engine->pitch >  1.5f) engine->pitch =  1.5f;
            if (engine->pitch < -1.5f) engine->pitch = -1.5f;
            engine->lastMouseX = xpos;
            engine->lastMouseY = ypos;
        }
    });

}

/**
 * @brief Runs the 3D engine,
 * Configure engine with camera position and field of view.
 * Plotting region is delimited by a cube : {._min, ._max} for (X, Y, Z)
 * Camera, cube center are alligned.
 * 
 * @param xmin min value of X
 * @param xmax max value of X
 * @param ymin min value of Y
 * @param ymax max value of Y
 * @param zmin min value of Z
 * @param zmax max value of Z
 * @param fovy_deg angle of view on Y axis 
 * @param k : Scaling factor for zoom out/in
 */
void Engine3D::run() {

    //center of cube of plot delimited by ._min & ._max values
    vec3 cube_center = {(this->xmin + this->xmax)/2,
                        (this->ymin + this->ymax)/2, 
                        (this->zmin + this->zmax)/2}; 

    //Radius of circle of center C that is arround the cube of plot 
    float Radius = 0.5f*sqrt(pow(this->xmax-this->xmin, 2)
                            +pow(this->ymax-this->ymin, 2)+pow(this->zmax-this->zmin, 2));

    //Distance bewteen circle(or cube) center and camera such that circle(C, R) is in field of view
    // Therefore camera is looking at what we plot
    float distance_camera_circleCenter = Radius/std::tan((this->fovy_deg/2)*(M_PI/180));
    distance_camera_circleCenter *= this->k ; //Zoom in/out

    // Initial camera position
    vec3 baseOffset = { distance_camera_circleCenter / sqrt(3), distance_camera_circleCenter / sqrt(3), distance_camera_circleCenter / sqrt(3) };

    // Rotation matrix (for mouse user rotation)
    Eigen::AngleAxisf rotY(yaw,   Eigen::Vector3f::UnitY());

    Eigen::AngleAxisf rotX(pitch, Eigen::Vector3f::UnitX());
    

    Eigen::Matrix3f rotMat = (rotX * rotY).toRotationMatrix();
    Eigen::Vector3f offset(baseOffset[0], baseOffset[1], baseOffset[2]);
    Eigen::Vector3f rotatedOffset = rotMat * offset;

    //Camera position = cube_center + (1, 1, 1)* distance_camera_circleCenter/sqrt(3) * Rot
    //Because Camera, cube center are alligned
    vec3 camera_pos = {cube_center[0] + rotatedOffset.x(),
                       cube_center[1] + rotatedOffset.y(),
                       cube_center[2] + rotatedOffset.z()};


    // Closest and further possible Z value to plot
    float zNear = std::max(0.01f, distance_camera_circleCenter - Radius) ; 
    float zFar =  distance_camera_circleCenter + Radius ;


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    // *** Sets projection parameters ***
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //Define "Pyramide of view" : angle, ratio, 
    gluPerspective(this->fovy_deg, (float)WIDTH/HEIGHT, zNear, zFar);

    // *** Sets point of view position ***
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(camera_pos[0], camera_pos[1] , camera_pos[2],  // camera position 
              cube_center[0], cube_center[1], cube_center[2],  // look-at point : origin (0, 0, 0) <-> cube center
              0.0, 0.0, 1.0); // Z is "upper" axis (Right-hand basis rule)
}

 /**    
* @brief Sets up camera position and field of view.
 * Plotting region is delimited by a cube : {._min, ._max} for (X, Y, Z)
 * Camera, cube center are alligned.
 * 
 * @param xmin min value of X
 * @param xmax max value of X
 * @param ymin min value of Y
 * @param ymax max value of Y
 * @param zmin min value of Z
 * @param zmax max value of Z
 * @param fovy_deg angle of view on Y axis 
 * @param k Scaling factor for zoom in/out */

void Engine3D::set_camera(float xmin, float xmax,
                    float ymin, float ymax,
                    float zmin, float zmax,
                    float fovy_deg, float k){
    this->xmin = xmin;
    this->xmax = xmax;
    this->ymin = ymin;
    this->ymax = ymax;
    this->zmin = zmin;
    this->zmax = zmax;
    this->fovy_deg = fovy_deg;
    this->k = k;
    
}

/**
 * @brief Sets a grid in 3D space on XY, XZ and YZ planes
 * @param res_x Distance bewteen two axes (YZ) on (X=0) axis 
 * @param res_y Distance bewteen two axes (XZ) on (Y=0) axis 
 * @param res_z Distance bewteen two axes (XY) on (Z=0) axis
 */
void Engine3D::set_grid_3D(float res_x, float res_y, float res_z) {

    //center of cube of plot delimited by ._min & ._max values
    vec3 cube_center = {(this->xmin + this->xmax)/2,
                        (this->ymin + this->ymax)/2, 
                        (this->zmin + this->zmax)/2}; 
    glLineWidth(0.2f);
    glBegin(GL_LINES);

    // --- Plan XY (z=zmin) ---
    glColor3f(0.7f, 0.7f, 0.7f);
    int i = 0;
    while (cube_center[0]+ i * res_x < this->xmax) {
        glVertex3f( cube_center[0]+ i * res_x, this->ymin, this->zmin);
        glVertex3f( cube_center[0]+ i * res_x, this->ymax, this->zmin);
        glVertex3f( cube_center[0]+-i * res_x, this->ymin, this->zmin);
        glVertex3f( cube_center[0]+-i * res_x, this->ymax, this->zmin);
        i++;
    }
    i = 0;
    while (cube_center[1]+i * res_y < this->ymax) {
        glVertex3f(this->xmin,cube_center[1]+  i * res_y, this->zmin);
        glVertex3f(this->xmax,cube_center[1]+  i * res_y, this->zmin);
        glVertex3f(this->xmin,cube_center[1]+ -i * res_y, this->zmin);
        glVertex3f(this->xmax,cube_center[1]+ -i * res_y, this->zmin);
        i++;
    }

    // --- Plan XZ (y=ymin) ---
    glColor3f(0.5f, 0.5f, 0.5f);
    i = 0;
    while (cube_center[0]+i * res_x < this->xmax) {
        glVertex3f(cube_center[0]+ i * res_x, this->ymin, this->zmin);
        glVertex3f(cube_center[0]+ i * res_x, this->ymin, this->zmax);
        glVertex3f(cube_center[0]+-i * res_x, this->ymin, this->zmin);
        glVertex3f(cube_center[0]+-i * res_x, this->ymin, this->zmax);
        i++;
    }
    i = 0;
    while (cube_center[2]+i * res_z < this->zmax) {
        glVertex3f(this->xmin, this->ymin, cube_center[2]+  i * res_z);
        glVertex3f(this->xmax, this->ymin, cube_center[2]+  i * res_z);
        glVertex3f(this->xmin, this->ymin, cube_center[2]+ -i * res_z);
        glVertex3f(this->xmax, this->ymin, cube_center[2]+ -i * res_z);
        i++;
    }

    // --- Plan YZ (x=xmin) ---
    glColor3f(0.5f, 0.5f, 0.5f);
    i = 0;
    while (cube_center[1]+i * res_y < this->ymax) {
        glVertex3f(this->xmin, cube_center[1]+  i * res_y, this->zmin);
        glVertex3f(this->xmin, cube_center[1]+  i * res_y, this->zmax);
        glVertex3f(this->xmin, cube_center[1]+ -i * res_y, this->zmin);
        glVertex3f(this->xmin, cube_center[1]+ -i * res_y, this->zmax);
        i++;
    }
    i = 0;
    while (cube_center[2]+i * res_z < this->zmax) {
        glVertex3f(this->xmin, this->ymin,cube_center[2]+  i * res_z);
        glVertex3f(this->xmin, this->ymax,cube_center[2]+  i * res_z);
        glVertex3f(this->xmin, this->ymin,cube_center[2]+ -i * res_z);
        glVertex3f(this->xmin, this->ymax,cube_center[2]+ -i * res_z);
        i++;
    }
    glEnd();

    // --- White ref. frame---
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 1.0f);
    glVertex3f(this->xmin, this->ymin, this->zmin); glVertex3f(this->xmax, this->ymin, this->zmin); // X
    glVertex3f(this->xmin, this->ymin, this->zmin); glVertex3f(this->xmin, this->ymax, this->zmin); // Y
    glVertex3f(this->xmin, this->ymin, this->zmin); glVertex3f(this->xmin, this->ymin, this->zmax); // Z

    glEnd();

    // --- Labels ---
    draw_text_3D(this->xmax, this->ymin, 0.05*(this->zmax - this->zmin)+this->zmin, "X");
    draw_text_3D(this->xmin, this->ymax, 0.05*(this->zmax - this->zmin)+this->zmin, "Y");
    draw_text_3D(this->xmin, this->ymin, this->zmax, "Z");
}



/**
 * @brief Plot point in 3D space
 * 
 * @param pos point coordinates
 * @param color {R, G, B} vector, max = 1.0
 */
void Engine3D::plot_couple_3D(vec3 pos, vec3 color) {
    //Radius of plotting area
    float Radius = 0.5f * sqrt(pow(this->xmax-this->xmin,2) + pow(this->ymax-this->ymin,2) + pow(this->zmax-this->zmin,2));
    float r = 0.002f * Radius; //points size is ratio r of Radius
    glColor3f(color[0], color[1], color[2]);
    glPushMatrix();
    glTranslatef(pos[0], pos[1], pos[2]);
    glutSolidSphere(r, 10, 10); 
    glPopMatrix();
}

/**
 * @brief Plot a surface for given (X, Y, Z) vectors
 * 
 * @param X x position
 * @param Y y position
 * @param Z z position
 * @param color {R, G, B} vector, max = 1.0
 */
void Engine3D::draw_surface_wireframe(const Eigen::VectorXf& X, 
                                      const Eigen::VectorXf& Y, 
                                      const Eigen::MatrixXf& Z,
                                      const vec3& color) {
    int m = X.size();
    int n = Y.size();

    glColor3f(color[0], color[1], color[2]);
    glLineWidth(0.3f);

    // Horizontal lines
    for (int i = 0; i < n; ++i) {
        glBegin(GL_LINE_STRIP);
        for (int j = 0; j < m; ++j) {
            glVertex3f(X(j), Y(i), Z(i, j));
        }
        glEnd();
    }
    // Vertical lines
    for (int j = 0; j < m; ++j) {
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < n; ++i) {
            glVertex3f(X(j), Y(i), Z(i, j));
        }
        glEnd();
    }
}

/**
 * @brief Draw text on (x, y, z) position
 * 
 * @param x x position
 * @param y y position
 * @param z z position
 * @param text string to be displayed
 */
void draw_text_3D(float x, float y, float z, const string& text) {
    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos3f(x, y, z);  // ← glRasterPos3f au lieu de glRasterPos2f
    for (char c : text) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
    }
}

