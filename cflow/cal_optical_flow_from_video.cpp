//
// Created by lshi on 3/12/18.
//

#include "list_file.h"
#include "cal_optical_flow.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, const char *argv[]) {

    const char *keys =
            {
                    "{ i img_dir  | /home/lshi/Database/UCF101/video/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi | filename of video }"
                            "{ f flow_dir  | /home/lshi/Database/UCF101Flow/UCF101FlowTVLVideo/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 | filename of flow x component }"
                            "{ b bound  | 15 | specify the maximum of optical flow}"
                            "{ t type  | 0 | specify the optical flow algorithm }"
                            "{ d device_id | 0  | set gpu id}"
                            "{ s step  | 2 | specify the step for frame sampling}"
            };

//    if(type==0){
//        cv::Ptr<cuda::OpticalFlowDual_TVL1> alg = cuda::OpticalFlowDual_TVL1::create();
//    } else if(type==1){
//        cv::Ptr<cuda::BroxOpticalFlow> alg = cuda::BroxOpticalFlow::create();
//    }else if(type==2){
//        cv::Ptr<cuda::FarnebackOpticalFlow> alg = cuda::FarnebackOpticalFlow::create();
//    }else if(type==3){
//        cv::Ptr<cuda::DensePyrLKOpticalFlow> alg = cuda::DensePyrLKOpticalFlow::create();
//    }

    CommandLineParser cmd(argc, argv, keys);
    string img_dir = cmd.get<string>("img_dir");
    string flow_dir = cmd.get<string>("flow_dir");

    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");
    int type = cmd.get<int>("type");
    int bound = cmd.get<int>("bound");

    cal_flow_gpu_from_video(img_dir, flow_dir, step, device_id, type, bound);
    return 0;
}