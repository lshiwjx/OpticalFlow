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
using namespace std;
using namespace cv;
using namespace cv::cuda;

void showGpuFlow(const char* name, const GpuMat& d_flow);

void showCpuFlow(const char* name, const Mat& d_flow);

void writeGpuFlow(const char* name, const GpuMat& d_flow);

#endif //OPTICALFLOW_CAL_OPTICAL_FLOW_H
