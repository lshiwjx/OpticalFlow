//
// Created by lshi on 3/16/18.
//
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "list_file.h"

//#include <iostream>
//#include <string>

//#include "list_file.h"
//#include "cal_optical_flow.h"
//#include "opencv2/cudawarping.hpp"
//#include <ctime>
using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, const char *argv[]) {

//    string img1 = "/home/lshi/Database/UCF101/img/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00001.jpg";
//    string img2 = "/home/lshi/Database/UCF101/img/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00005.jpg";
    string video_path1 =  "/home/lshi/Project/caffe2/interesting/interesting_torch/test/video/963193352.mp4";
    string video_path2 = "/home/lshi/Database/meitu/test_video/video/231125424.mp4";
    clock_t s;
    clock_t e;
    Mat outimg;
    GpuMat gpumat;
    string root = "/home/lshi/Database/meitu/test_video/video/";
    vector<string> cls = getFiles(root);

    cv::Ptr<cudacodec::VideoReader> video_stream = cudacodec::createVideoReader(video_path2);
    video_stream->nextFrame(gpumat);
    VideoCapture video = VideoCapture(video_path2, VideoCaptureAPIs::CAP_FFMPEG);
    video>>outimg;

    for (int i = 0; i < 5; ++i) {

        string video_path = root + '/' + cls[i];
        s = clock();
        cv::Ptr<cudacodec::VideoReader> video_stream = cudacodec::createVideoReader(video_path1);
        video_stream->nextFrame(gpumat);
        e = clock();
        cout<<double(e-s)/1000<<std::endl;

        s = clock();
        for (int j = 0; j < 100; ++j) {
            video_stream->nextFrame(gpumat);
        }
        video_stream->nextFrame(gpumat);
        e = clock();
        cout<<double(e-s)/1000<<std::endl;
//        cudacodec::ChromaFormat c = vide
//        cudacodec::ChromaFormat c = video_stream->format().chromaFormat;
//        cudacodec::Codec codec = video_stream->format().codec;
//        cout << c << codec << endl;


        s = clock();
        video.open(video_path1, VideoCaptureAPIs::CAP_FFMPEG);
        video>>outimg;
        e = clock();
        cout<<double(e-s)/1000<<std::endl;

        s = clock();
        video>>outimg;
        e = clock();
        cout<<double(e-s)/1000<<std::endl;
    }

//    int h = video_stream->format().height;
//    int w = video_stream->format().width;
//    GpuMat capture_frame, resize_frame(200, 300, CV_8UC4);

//
//    s = clock();
//    cv::Ptr<cudacodec::VideoReader> video_stream3 = cudacodec::createVideoReader(video_path2);
//    e = clock();
//    cout<<double(e-s)/1000<<std::endl;
//    s = clock();
//    cv::Ptr<cudacodec::VideoReader> video_stream2 = cudacodec::createVideoReader(video_path1);
//    e = clock();
//    cout<<double(e-s)/1000<<std::endl;
//    cudacodec::ChromaFormat c = video_stream->format().chromaFormat;
//    cudacodec::Codec codec = video_stream->format().codec;
//    for (int j = 0; j < 60; ++j) {
//        video_stream->nextFrame(capture_frame);  //BGRA
//    }
//    s = clock();
//    video_stream->nextFrame(capture_frame);
//    int channel = capture_frame.channels();
//    capture_frame.download(outimg);
//    cv::imshow("t", outimg);
//    cvWaitKey(0);

//    cv::cuda::cvtColor(capture_frame, resize_frame, CV_RGBA2BGR);
//
//    resize_frame.download(outimg);
//    cv::imshow("t", outimg);
//    cvWaitKey(0);

//    cout << h << w << c << codec << endl;
//    e = clock();
//    cout<<double(e-s)/1000<<std::endl;

//    for (int i = 0; i < 2; ++i) {
//        s = clock();
//        cv::Ptr<cudacodec::VideoReader> video_stream1 = cudacodec::createVideoReader(video_path2);
//        video_stream1->nextFrame(capture_frame);
//        cuda::resize(capture_frame, resize_frame, cv::Size(200,300));
//        e = clock();
//        cout<<double(e-s)/1000<<std::endl;
//    }
//    s = clock();
//    cv::Ptr<cudacodec::VideoReader> video_stream1 = cudacodec::createVideoReader(video_path2);
//    for (int i = 0; i < 60; ++i) {
//        video_stream1->nextFrame(capture_frame);
//    }
//    for (int i = 0; i < 100; ++i) {
//        video_stream1->nextFrame(capture_frame);
////        cuda::resize(capture_frame, resize_frame, cv::Size(50,400));
////        size_t e = resize_frame.elemSize();
////        e = resize_frame.elemSize1();
//
////        cout << resize_frame.channels() << endl;
////        resize_frame.download(outimg);
////        cv::imshow("t", outimg);
////        cvWaitKey(0);
//    }
//    e = clock();
//    cout<<double(e-s)/1000<<std::endl;
    return 0;
}