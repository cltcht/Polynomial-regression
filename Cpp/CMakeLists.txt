cmake_minimum_required(VERSION 3.16)

project(Plot LANGUAGES CXX)

set(exec regression)
set(MAIN_FILE regression_polynomiale.cpp)

# ---- Standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Optimization ----
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# ---- Find packages ----
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLUT REQUIRED)

# ---- Collect source ----
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/plot_graph.cpp
    ${CMAKE_SOURCE_DIR}/src/plot_graph_3D.cpp
    ${CMAKE_SOURCE_DIR}/src/data_setp.cpp
    ${CMAKE_SOURCE_DIR}/src/${MAIN_FILE}
)

# ---- include ----
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---- Create executable ----
add_executable(${exec} ${SOURCES})

# ---- Link libraries ----
target_link_libraries(${exec} PRIVATE
    OpenGL::GL
    OpenGL::GLU
    glfw
    GLEW::GLEW
    Eigen3::Eigen
    GLUT::GLUT  
)
