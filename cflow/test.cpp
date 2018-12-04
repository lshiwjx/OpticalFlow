//
// Created by lshi on 12/4/18.
//

//
// Created by lshi on 3/12/18.
//

#include "cal_optical_flow.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, const char *argv[]) {

    string img1 = "../data/00001.jpg";
    string img2 = "../data/00005.jpg";
    string video = "../data/v_ApplyEyeMakeup_g01_c01.avi";
    string out = "../out/";

    cal_flow_gpu_from_video(video, out);
    compare_gpu_cpu_flow(img1, img2);
    compare_flow_methods(img1, img2);
    return 0;
}