//
// Created by lshi on 3/16/18.
//

#include "list_file.h"
#include "cal_optical_flow.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, const char *argv[]) {

    string img1 = "/home/lshi/Database/UCF101/img/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00001.jpg";
    string img2 = "/home/lshi/Database/UCF101/img/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00005.jpg";

    compare_gpu_cpu_flow(img1, img2);
    return 0;
}