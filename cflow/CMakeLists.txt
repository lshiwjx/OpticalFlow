#set(OpenCV_DIR /home/lshi/Application/Anaconda/share/OpenCV/)

find_package(OpenCV REQUIRED)

message(${CMAKE_MODULE_PATH})
message(${OpenCV_INCLUDE_DIRS})
#message(${OpenCV_LIBRARIES_DIRS})
message(${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(
##        "/home/lshi/Application/Anaconda/envs/pytorch/lib/python3.6/site-packages/torch/lib/include"
#        "/usr/local/cuda/include"
#        ".."
#)

set(SOURCE_FILES_VIDEO cal_optical_flow_from_video.cpp cal_optical_flow.cpp list_file.cpp)
set(SOURCE_FILES_IMG cal_optical_flow_from_img.cpp cal_optical_flow.cpp list_file.cpp mkdir.cpp mkdir.h)
set(SOURCE_FILES_TEST test.cpp cal_optical_flow.cpp list_file.cpp)

add_executable(calOpticalFlowFromImg ${SOURCE_FILES_IMG})
add_executable(calOpticalFlowFromVideo ${SOURCE_FILES_VIDEO})
add_executable(test ${SOURCE_FILES_TEST})

target_link_libraries(calOpticalFlowFromImg ${OpenCV_LIBS})
target_link_libraries(calOpticalFlowFromVideo ${OpenCV_LIBS})
target_link_libraries(test ${OpenCV_LIBS})