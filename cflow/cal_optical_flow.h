//
// Created by lshi on 3/1/18.
//

#ifndef OPTICALFLOW_CAL_OPTICAL_FLOW_H
#define OPTICALFLOW_CAL_OPTICAL_FLOW_H

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "list_file.h"
#include "unistd.h"
#include "sstream"
#include "iomanip"

using namespace std;
using namespace cv;
using namespace cv::cuda;

void showGpuFlow(const char* name, const GpuMat& d_flow);

void showCpuFlow(const char* name, const Mat& d_flow);

//calculate the optical flow from images given a dir list
void cal_flow_from_dir_list(vector<string> dir_list, vector<string> flow_list, int step=1, int dev=0);

//calculate the optical flow from images in a given dir
void cal_flow_from_dir(string img_dir, string flow_dir, int step, int device_id);

void compare_gpu_cpu_flow(string img1, string img2);

void compare_flow_methods(string img1, string img2);

// compute and save the optical flow image in a dir
void cal_flow_gpu_from_video(string video_path, string flow_dir, int step = 1, int dev_id = 0, int type = 0, int bound = 10);

#endif //OPTICALFLOW_CAL_OPTICAL_FLOW_H
