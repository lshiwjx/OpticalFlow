#include <torch/torch.h>
#include <string>
//#include "opencv2/core.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudawarping.hpp"
//#include "opencv2/cudawarping.hpp"
#include <ctime>
#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::cuda;


void video_capture_resize_crop(string video_path, at::Tensor &tensor, at::Tensor &ftensor,
                               int minsize = 120, int cropsize = 112, int length = 1,
                               int skip = 0, int internal = 1, int pw = 0, int ph = 0) {
    int max_h = tensor.size(1);
    int max_w = tensor.size(2);
    assert(tensor.size(0) >= length);
    assert(tensor.size(3) == 4);
    assert(tensor.type() == torch::CUDA(at::kByte));

    assert(internal >= 1);
    assert(pw < 3);
    assert(ph < 3);

    assert(ftensor.size(0) == length);
    assert(ftensor.size(1) == cropsize);
    assert(ftensor.size(2) == cropsize);
    assert(ftensor.size(3) == 4);
    assert(ftensor.type() == torch::CUDA(at::kByte));

    cv::Ptr<cudacodec::VideoReader> video_stream = cudacodec::createVideoReader(video_path);

    int h = video_stream->format().height;
    int w = video_stream->format().width;
    cudacodec::Codec c = video_stream->format().codec;
    cudacodec::ChromaFormat codec = video_stream->format().chromaFormat;
//    cout << h << w << c << codec << endl;

    assert(max_h >= h);
    assert(max_w >= w);

    //resize
    int new_h, new_w;
    if (h > w) {
        new_w = minsize;
        new_h = int(h / w * new_w);
    } else {
        new_h = minsize;
        new_w = int(w / h * new_h);
    }
//    cout << new_h << new_w << endl;

    //crop
    int pwa[3] = {0, (new_w - cropsize) / 2, new_w - cropsize};
    int pha[3] = {0, (new_h - cropsize) / 2, new_h - cropsize};
    Rect roi(pwa[pw], pha[ph], cropsize, cropsize);
//    cout << roi << endl;

    GpuMat capture_frame(1000, 1000, CV_8UC4, (char *) tensor.data_ptr());
    GpuMat final_frame(cropsize, cropsize, CV_8UC4, (char *) ftensor.data_ptr());

    for (int i = 0; i < skip; ++i) {
        if (!video_stream->nextFrame(capture_frame)) break;
    }

    for (int j = 0; j < length; ++j) {
        for (int i = 0; i < internal; ++i) {
            if (!video_stream->nextFrame(capture_frame)) break;
        }
        cuda::resize(capture_frame, capture_frame, cv::Size(new_w, new_h));

//        cout << capture_frame(roi).isContinuous() << endl;

        capture_frame(roi).convertTo(final_frame, final_frame.type());

    }
}


at::Tensor video_capture_resize(string video_path, int minsize = 120, int length = 1, int skip = 0, int internal = 1) {
//    clock_t s = clock();
    cv::Ptr<cudacodec::VideoReader> video_stream = cudacodec::createVideoReader(video_path);
    int h = video_stream->format().height;
    int w = video_stream->format().width;
    cudacodec::Codec c = video_stream->format().codec;
    cudacodec::ChromaFormat codec = video_stream->format().chromaFormat;
    cout << h << w << c << codec << endl;
    int new_h, new_w;
    if (h > w) {
        new_w = minsize;
        new_h = int(h / w * new_w);
    } else {
        new_h = minsize;
        new_w = int(w / h * new_h);
    }
//    cout<<double(clock()-s)<<std::endl;
//    s = clock();
//    cout << pointer << endl;
//    void *pointer_ = reinterpret_cast<void *>(pointer);
//    at::Tensor tensor = at::CUDA(at::kByte).zeros({w, h, 4});
    at::Tensor tensor = at::ones(torch::CUDA(at::kByte), {length, new_w, new_h, 4});
//    GpuMat capture_frame, resize_frame(w, h, CV_8UC4, (uchar *) tensor.data_ptr());
//    tensor.resize_();
//    cv::Mat x_cpu(w, h, CV_32FC3);
//    x_cpu = cv::Scalar::all(1);
//    capture_frame.upload(x_cpu);

//    cout << capture_frame.size() << capture_frame.channels() << endl;
//    int ret = video_stream->nextFrame(capture_frame);
//    cout << ret << endl;
//    cuda::resize(capture_frame, resize_frame, cv::Size(new_w, new_h));
//    cout<<double(clock()-s)<<std::endl;
//    s = clock();
    GpuMat capture_frame;
    for (int i = 0; i < skip; ++i) {
        if (!video_stream->nextFrame(capture_frame)) return tensor;
    }

    for (int j = 0; j < length; ++j) {
        GpuMat resize_frame(new_w, new_h, CV_8UC4, (uchar *) tensor[j].data_ptr());
        for (int i = 0; i < internal; ++i) {
            if (!video_stream->nextFrame(capture_frame)) return tensor;
        }
        cuda::resize(capture_frame, resize_frame, cv::Size(new_w, new_h));
    }
//    cout<<double(clock()-s)<<std::endl;
//    cout << *capture_frame.datastart << endl;
//    video_stream1->nextFrame(capture_frame);
//    int height = capture_frame->size().height;
//    int width = capture_frame->size().width;
//    int size[2] = {height, width};
//    cout << height << width << endl;
//    at::Tensor tensor=at::CUDA(at::kChar).zero_({height,width, 3});
    return tensor;
}


at::Tensor video_capture(string video_path, int length = 1, int skip = 0, int internal = 1) {
//    clock_t s = clock();
    cv::Ptr<cudacodec::VideoReader> video_stream = cudacodec::createVideoReader(video_path);
    int h = video_stream->format().height;
    int w = video_stream->format().width;
    cudacodec::Codec c = video_stream->format().codec;
    cudacodec::ChromaFormat codec = video_stream->format().chromaFormat;
    cout << h << w << c << codec << endl;

//    cout<<double(clock()-s)<<std::endl;
//    s = clock();
//    cout << pointer << endl;
//    void *pointer_ = reinterpret_cast<void *>(pointer);
//    at::Tensor tensor = at::CUDA(at::kByte).zeros({w, h, 4});
    at::Tensor tensor = at::ones(torch::CUDA(at::kByte), {length, w, h, 4});
//    GpuMat capture_frame, resize_frame(w, h, CV_8UC4, (uchar *) tensor.data_ptr());
//    tensor.resize_();
//    cv::Mat x_cpu(w, h, CV_32FC3);
//    x_cpu = cv::Scalar::all(1);
//    capture_frame.upload(x_cpu);

//    cout << capture_frame.size() << capture_frame.channels() << endl;
//    int ret = video_stream->nextFrame(capture_frame);
//    cout << ret << endl;
//    cuda::resize(capture_frame, resize_frame, cv::Size(new_w, new_h));
//    cout<<double(clock()-s)<<std::endl;
//    s = clock();
    GpuMat capture_frame(w, h, CV_8UC4, (uchar *) tensor.data_ptr());
    for (int i = 0; i < skip; ++i) {
        if (!video_stream->nextFrame(capture_frame)) return tensor;
    }

    for (int j = 0; j < length; ++j) {
        GpuMat resize_frame(w, h, CV_8UC4, (uchar *) tensor[j].data_ptr());
        for (int i = 0; i < internal; ++i) {
            if (!video_stream->nextFrame(capture_frame)) return tensor;
        }
    }
//    cout<<double(clock()-s)<<std::endl;
//    cout << *capture_frame.datastart << endl;
//    video_stream1->nextFrame(capture_frame);
//    int height = capture_frame->size().height;
//    int width = capture_frame->size().width;
//    int size[2] = {height, width};
//    cout << height << width << endl;
//    at::Tensor tensor=at::CUDA(at::kChar).zero_({height,width, 3});
    return tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("video_capture", &video_capture, "video_capture");
    m.def("video_capture_resize", &video_capture_resize, "video_capture_resize");
    m.def("video_capture_resize_crop", &video_capture_resize_crop, "video_capture_resize_crop");
}