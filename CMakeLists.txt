CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(reconstruction)

SET(g2o_DIR /home/stwirth/src/g2o)

# Set search directory for looking for our custom CMake scripts to
# look for SuiteSparse, QGLViewer, and Eigen3.
LIST(APPEND CMAKE_MODULE_PATH ${g2o_DIR}/cmake_modules)

FIND_PACKAGE(OpenCV REQUIRED)

# For building the CHOLMOD / CSPARSE solvers
FIND_PACKAGE(Cholmod)
FIND_PACKAGE(BLAS)
FIND_PACKAGE(LAPACK)
IF(CHOLMOD_FOUND AND BLAS_FOUND AND LAPACK_FOUND)
  MESSAGE(STATUS "Found CHOLMOD and its dependencies")
  SET(CHOLMOD_FOUND TRUE)
ELSE()
  SET(CHOLMOD_FOUND FALSE)
ENDIF()

FIND_PACKAGE(CSparse)

# Compiler specific options for gcc
IF(CMAKE_COMPILER_IS_GNUCXX)
  MESSAGE(STATUS "Compiling with GCC")
  
  # Generic settings for optimisation
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse4") 
  SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -msse4") 
  # switch off optimization for debug builds
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

  # OS X
  #IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    #SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") 
    #SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}") 
  #ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Linux
  IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native") 
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native") 
  ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# Find Eigen3
SET(EIGEN3_INCLUDE_DIR ${G2O_EIGEN3_INCLUDE})
FIND_PACKAGE(Eigen3 REQUIRED)

# Set up the top-level include directories
INCLUDE_DIRECTORIES(${g2o_DIR} ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

add_executable(sba src/sba.cpp)
# Link CHOLMOD if available, CSparse otherwise
IF(CHOLMOD_FOUND)
  INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(sba g2o_solver_cholmod)
ELSE()
  INCLUDE_DIRECTORIES(${CSPARSE_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(sba g2o_solver_csparse)
ENDIF()

TARGET_LINK_LIBRARIES(sba -L${g2o_DIR}/lib g2o_core g2o_stuff g2o_types_icp g2o_types_slam3d cholmod)

TARGET_LINK_LIBRARIES(sba opencv_core opencv_features2d)

