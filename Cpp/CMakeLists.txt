cmake_minimum_required(VERSION 3.16)

project(Plot LANGUAGES CXX)

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

# ---- Shared sources ----
set(SHARED_SOURCES
    ${CMAKE_SOURCE_DIR}/src/plot_graph.cpp
    ${CMAKE_SOURCE_DIR}/src/plot_graph_3D.cpp
    ${CMAKE_SOURCE_DIR}/src/data_setp.cpp
)

# ---- Main files ----
set(MAIN_FILES
    ${CMAKE_SOURCE_DIR}/src/Linear_regression_MSE_matrix_inversion.cpp
    ${CMAKE_SOURCE_DIR}/src/Multi_Polynomial_regression_MSE_matrix_inversion.cpp
    ${CMAKE_SOURCE_DIR}/src/Polynomial_regression_MSE_matrix_inversion.cpp
)

# ---- Include ----
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---- One executable per main file ----
foreach(main_file ${MAIN_FILES})

    get_filename_component(exec_name ${main_file} NAME_WE)

    add_executable(${exec_name}
        ${main_file}
        ${SHARED_SOURCES}
    )

    target_link_libraries(${exec_name} PRIVATE
        OpenGL::GL
        OpenGL::GLU
        glfw
        GLEW::GLEW
        Eigen3::Eigen
        GLUT::GLUT
    )

endforeach()