cmake_minimum_required(VERSION 3.10)

# set the project name
project(CSI_Camera_Calibration)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-fPIC -g -pedantic -Wall -Wextra")

if(LOG) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOG=1")
endif()

# make sure we have the necessary submodule
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/CSI_Camera/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

find_package(OpenCV 4.5 REQUIRED)

# add OpenCV directory
include_directories(${OpenCV_INCLUDE_DIR})
# add the main source directory
include_directories(src)

# add the CSI Camera submodule
add_subdirectory(CSI_Camera)
include_directories(CSI_Camera/src)
include_directories(CSI_Camera/JetracerUtils/src)

# Build the calibration tool
add_executable(CSI_Camera_Calibration src/Calibration.cpp src/StereoListener.cpp src/main.cpp)
target_link_libraries(CSI_Camera_Calibration CSI_Camera ${OpenCV_LIBRARIES})

