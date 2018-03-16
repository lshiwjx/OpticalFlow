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
void cal_flow_from_list(vector<string> dir_list, vector<string> flow_list, int step, int dev);
void writeFlowImg(string name, const GpuMat& d_flow);

void showGpuFlow(const char* name, const GpuMat& d_flow);

void showCpuFlow(const char* name, const Mat& d_flow);

void writeGpuFlow(const char* name, const GpuMat& d_flow);

void compute_and_save_ucf_flow(string root, string flow_root);

void compute_and_save_twbn_flow(string root, string flow_root);

void compute_and_save_dir_flow(string img_dir, string flow_dir, int step, int device_id);

void compare_gpu_cpu_flow(string img1, string img2);

void compare_flow_methods(string img1, string img2);

void calcDenseFlowPureGPU(string video_path, string flow_dir, int step, int dev_id);

#endif //OPTICALFLOW_CAL_OPTICAL_FLOW_H
