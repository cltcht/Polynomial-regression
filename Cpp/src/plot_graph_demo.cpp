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
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;

vec2 mouseWorld(0.0f);
struct Engine {

    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;

    Engine (const char* title) {
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
    void run() {
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        // set origin to centre
        double halfWidth = WIDTH / 2.0f, halfHeight = HEIGHT / 2.0f;
        glOrtho(-halfWidth, halfWidth, -halfHeight, halfHeight, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
    void set_grid(){

        float res_h = 50.0f;
        float res_v = 50.0f;
        float nb_lignes_h = HEIGHT/res_h;
        float nb_lignes_v = WIDTH/res_v;

        
        glBegin(GL_LINES);
        glLineWidth(0.1f);
        glColor3f(0.5f, 0.5f, 0.5f);
        // lignes horizontales
        for (int i = 0; i <= nb_lignes_h; i++) {
            float y = i * res_h - HEIGHT/2.0f;
            glVertex2f(-WIDTH/2.0f, y); // start
            glVertex2f(WIDTH/2.0f, y);  // end
        }

        // lignes verticales
        for (int i = 0; i <= nb_lignes_v; i++) {
            float x = i * res_v - WIDTH/2.0f;
            glVertex2f(x, -HEIGHT/2.0f); // start
            glVertex2f(x, HEIGHT/2.0f);  // end
        }

        //Repère central
        glLineWidth(1.0f);
        glColor3f(1.0f, 1.0f, 1.0f);
        glVertex2f(0.0f, -HEIGHT/2.0f); // start
        glVertex2f(0.0f, HEIGHT/2.0f);  // end
        glVertex2f(-WIDTH/2.0f, 0.0f); // start
        glVertex2f(WIDTH/2.0f, 0.0f);  // end

        glEnd();

    
    }


};
Engine engine("Plot graph");

struct dataset{
    vector<int> X ;
    vector<int> Y ;
    vector<vec2> XY;

    dataset(vector<int>& vec_X, vector<int>& vec_Y) : X(vec_X), Y(vec_Y) 
    {

        if (vec_X.size() != vec_Y.size())
        {
            cerr << "vectors don't have the same size";
            exit(EXIT_FAILURE);
        }
        else 
        {
            for(int i = 0; i < vec_X.size(); i++){

                XY.emplace_back(X[i], Y[i]);
            }

        }
                               
    }
    void print_data(){
        printf("i | x ; y\n");
        for(int i = 0; i < X.size(); i++){
            printf("%d | %d ; %d\n", i, X[i], Y[i]);
        }
    }
};

// --- main ---
int main () {

    // Seed the random number generator using the current time (only once per program run)
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    // Definition points :
    int n = 10; // nombre de points
    int WIDTH = 800, HEIGHT = 600;
    int xmax = WIDTH/2, ymax = HEIGHT/2;
    int xmin = -xmax, ymin = -ymax;
    vector<int> X = { }; 
    vector<int> Y = { };

    for (int i = 0; i <= n; i++) {
        int random_x = (std::rand())%(xmax-xmin) + xmin;
        int random_y = (std::rand())%(ymax-ymin) + ymin;
        X.emplace_back(random_x);
        Y.emplace_back(random_y);
    }

    dataset data_couple(X, Y);

    data_couple.print_data();

    

    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();
        engine.set_grid();
        

        // --- Draw Points ---

        
        glBegin(GL_POINTS);
        glColor3f(1.0f, 0.0f, 0.0f);
        for(vec2 &xy : data_couple.XY)
            glVertex2f(xy[0], xy[1]);  // coordonnées monde (centrées chez toi)
        glEnd();
        


        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
} 