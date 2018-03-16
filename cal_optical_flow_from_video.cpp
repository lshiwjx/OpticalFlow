//
// Created by lshi on 3/12/18.
//

#include "list_file.h"
#include "cal_optical_flow.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, const char *argv[]) {

    const char* keys =
            {
                    "{ i img_dir  | /home/lshi/Database/UCF101/video/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi | filename of video }"
                            "{ f flow_dir  | /home/lshi/Database/UCF101Flow/UCF101FlowTVLVideo/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 | filename of flow x component }"
//                    "{ b  | bound | 15 | specify the maximum of optical flow}"
//                    "{ t  | type | 0 | specify the optical flow algorithm }"
                            "{ d device_id | 0  | set gpu id}"
                            "{ s step  | 2 | specify the step for frame sampling}"
            };

    CommandLineParser cmd(argc, argv, keys);
    string img_dir = cmd.get<string>("img_dir");
    string flow_dir = cmd.get<string>("flow_dir");

    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");
//    string root = "/home/lshi/Database/UCF101/";
//    string flow_root = "/home/lshi/Database/UCF101FlowTVL/";

//    string root = "/opt/Jester/20bn-jester-v1/";
//    string flow_root = "/home/lshi/Database/JesterFlow/TVL/";

//    string img1 = "/home/lshi/Database/UCF101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00001.jpg";
//    string img2 = "/home/lshi/Database/UCF101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00005.jpg";

//    compare_flow_methods(img1, img2);
    calcDenseFlowPureGPU(img_dir, flow_dir, step, device_id);
    return 0;
}