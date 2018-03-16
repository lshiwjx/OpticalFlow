//
// Created by lshi on 3/1/18.
//

#include <cv.hpp>
#include "cal_optical_flow.h"


inline bool isFlowCorrect(Point2f u) {
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

Vec3b computeColor(float fx, float fy) {
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first) {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++) {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

void drawOpticalFlow(const Mat_<float> &flowx, const Mat_<float> &flowy, Mat &dst, float maxmotion = -1) {
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0) {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y) {
            for (int x = 0; x < flowx.cols; ++x) {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y) {
        for (int x = 0; x < flowx.cols; ++x) {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

void showGpuFlow(const char *name, const GpuMat &d_flow) {
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    imshow(name, out);
}


void writeFlowImg(string name, const GpuMat &d_flow) {
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    if (!imwrite(name, out))
        cout << "wrong save";
}

void showCpuFlow(const char *name, const Mat &d_flow) {
    Mat planes[2];
    cv::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    imshow(name, out);
}

void writeGpuFlow(const char *name, const GpuMat &d_flow) {
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    cv::imwrite(name, out);
}


void cal_flow_from_list(vector<string> dir_list, vector<string> flow_list, int step, int dev) {
    Ptr<cuda::BroxOpticalFlow> flow = cuda::BroxOpticalFlow::create();
    setDevice(dev);
    string filename1, filename2;
    Mat frame0, frame1, out, flow_x, flow_y;
    GpuMat d_frame0, d_frame0f;
    GpuMat d_frame1, d_frame1f;
    GpuMat d_flow, planes[2];

    for (int i = 0; i < dir_list.size(); ++i) {
        int initial = 0;
        const int64 start = getTickCount();
        vector<string> imgs = getFiles(dir_list[i]);
        for (int j = 0; j < imgs.size(); j = j + step) {
            if (j < imgs.size() - step) {
                string flow_name = flow_list[i] + "/" + imgs[j + step];
                if ((access(flow_name.c_str(), 0)) != -1) {
                    cout << flow_name << " has exists" << endl;
                    continue;
                }
                if (initial == 0) {
                    filename1 = dir_list[i] + "/" + imgs[j];
                    frame0 = imread(filename1, IMREAD_GRAYSCALE);
                    initial = 1;
                } else {
                    frame0 = frame1;
                }

                filename2 = dir_list[i] + "/" + imgs[j + step];
                frame1 = imread(filename2, IMREAD_GRAYSCALE);

                if (frame0.empty()) {
                    cerr << "Can't open image [" << filename1 << "]" << endl;
                    continue;
                }
                if (frame1.empty()) {
                    cerr << "Can't open image [" << filename2 << "]" << endl;
                    continue;
                }

                if (frame1.size() != frame0.size()) {
                    cerr << "Images should be of equal sizes" << endl;
                    continue;
                }

                d_frame0.upload(frame0);
                d_frame1.upload(frame1);

                d_flow.create(frame0.size(), CV_32FC2);
                d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
                d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
                flow->calc(d_frame0f, d_frame1f, d_flow);

                cuda::split(d_flow, planes);

                planes[0].download(flow_x);
                planes[1].download(flow_y);

                drawOpticalFlow(flow_x, flow_y, out, 10);

                if (!imwrite(flow_name, out))
                    cout << "wrong save";
            }
        }

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << dir_list[i] << " : " << timeSec << " sec " << i << "/" << dir_list.size() << endl;

    }
}


void compute_and_save_dir_flow(string img_dir, string flow_dir, int step = 1, int device_id = 0) {
    vector<string> jpgs = getFiles(img_dir);
    string filename1, filename2;
//    int flag = 0;
//    for (int i = 0; i < jpgs.size(); i = i + step) {
//        if (i < jpgs.size() - step) {
//            string flow_name = flow_dir + "/" + jpgs[i + step];
//            if ((access(flow_name.c_str(), 0)) != -1) {
//                cout << flow_name << " has exists" << endl;
//                continue;
//            } else{
//                flag=1;
//                break;
//            }
//        }
//    }
//    if(flag==0) {
//        cout << img_dir << " reture" <<endl;
//        return;
//    }
    cuda::setDevice(device_id);
    cout << "create" << endl;
    Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();
    Mat frame0, frame1, flow_x, flow_y, out;
    GpuMat d_frame0, d_frame1, planes[2], d_flow;
    bool initial = 0;
    for (int j = 0; j < jpgs.size(); j = j + step) {
        if (j < jpgs.size() - step) {
            string flow_name = flow_dir + "/" + jpgs[j + step];
            if ((access(flow_name.c_str(), 0)) != -1) {
                cout << flow_name << " has exists" << endl;
                continue;
            }
            if (initial == 0) {
                filename1 = img_dir + "/" + jpgs[j];
                frame0 = imread(filename1, IMREAD_GRAYSCALE);
                initial = 1;
            } else {
                frame0 = frame1;
            }
            filename2 = img_dir + "/" + jpgs[j + step];
            frame1 = imread(filename2, IMREAD_GRAYSCALE);

            if (frame0.empty()) {
                cerr << "Can't open image [" << filename1 << "]" << endl;
                continue;
            }
            if (frame1.empty()) {
                cerr << "Can't open image [" << filename2 << "]" << endl;
                continue;
            }

            if (frame1.size() != frame0.size()) {
                cerr << "Images should be of equal sizes" << endl;
                continue;
            }

            d_frame0.upload(frame0);
            d_frame1.upload(frame1);

//            GpuMat d_flow(frame0.size(), CV_32FC2);
            d_flow.create(frame0.size(), CV_32FC2);
            const int64 start = getTickCount();
            tvl1->calc(d_frame0, d_frame1, d_flow);

//            GpuMat planes[2];
            cuda::split(d_flow, planes);

//            Mat flowx(planes[0]);
//            Mat flowy(planes[1]);
            planes[0].download(flow_x);
            planes[1].download(flow_y);

//            Mat out;
            drawOpticalFlow(flow_x, flow_y, out, 10);

            if (!imwrite(flow_name, out))
                cout << "wrong save";

//            writeFlowImg(flow_name, d_flow);
            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << flow_name << " : " << timeSec << " sec" << endl;
        }


    }
}


void compare_gpu_cpu_flow(string img1, string img2) {
    Ptr<cuda::FarnebackOpticalFlow> tvl1 = cuda::FarnebackOpticalFlow::create();
    Ptr<cv::FarnebackOpticalFlow> tvl2 = cv::FarnebackOpticalFlow::create();
    Mat frame0 = imread(img1, IMREAD_GRAYSCALE);
    Mat frame1 = imread(img2, IMREAD_GRAYSCALE);
    GpuMat d_frame0;
    GpuMat d_frame1;
    GpuMat d_flow(frame0.size(), CV_32FC2);
    Mat frame2 = imread(img1, IMREAD_GRAYSCALE);
    Mat frame3 = imread(img2, IMREAD_GRAYSCALE);
    Mat flow(frame2.size(), CV_32FC2);

    const int64 start = getTickCount();

    d_frame0.upload(frame0);
    d_frame1.upload(frame1);
    tvl1->calc(d_frame0, d_frame1, d_flow);
    const double timeSec1 = (getTickCount() - start) / getTickFrequency();
    cout << "TVLgpu : " << timeSec1 << " sec" << endl;

    const int64 start1 = getTickCount();
    tvl2->calc(frame2, frame3, flow);
    const double timeSec2 = (getTickCount() - start1) / getTickFrequency();
    cout << "TVLcpu : " << timeSec2 << " sec" << endl;

    showGpuFlow("TVLgpu", d_flow);
    showCpuFlow("TVLcpu", flow);
    waitKey();

}


void compare_flow_methods(string img1, string img2) {
    Mat frame0 = imread(img1, IMREAD_GRAYSCALE);
    Mat frame1 = imread(img2, IMREAD_GRAYSCALE);
    GpuMat d_frame0(frame0);
    GpuMat d_frame1(frame1);
    GpuMat d_flow(frame0.size(), CV_32FC2);

    Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
    Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();
    Ptr<cuda::OpticalFlowDual_TVL1> tvl = cuda::OpticalFlowDual_TVL1::create();

    {
        GpuMat d_frame0f;
        GpuMat d_frame1f;

        d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
        d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

        const int64 start = getTickCount();

        brox->calc(d_frame0f, d_frame1f, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Brox : " << timeSec << " sec" << endl;

        showGpuFlow("Brox", d_flow);
    }

    {
        const int64 start = getTickCount();

        lk->calc(d_frame0, d_frame1, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "LK : " << timeSec << " sec" << endl;

        showGpuFlow("LK", d_flow);
    }

    {
        const int64 start = getTickCount();

        farn->calc(d_frame0, d_frame1, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Farn : " << timeSec << " sec" << endl;

        showGpuFlow("Farn", d_flow);
    }

    {
        const int64 start = getTickCount();

        tvl->calc(d_frame0, d_frame1, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "TVL : " << timeSec << " sec" << endl;

        showGpuFlow("TVL", d_flow);
    }

    waitKey();
}


void calcDenseFlowPureGPU(string video_path, string flow_dir, int step = 1, int dev_id = 0) {

    setDevice(dev_id);
    cv::Ptr<cudacodec::VideoReader> video_stream = cudacodec::createVideoReader(video_path);

    GpuMat capture_frame, capture_image, prev_image, capture_gray, prev_gray, planes[2];
    Mat flow_x, flow_y, out;
    GpuMat d_frame0f;
    GpuMat d_frame1f;
    GpuMat d_flow;

    cv::Ptr<cuda::BroxOpticalFlow> alg_tvl1 = cuda::BroxOpticalFlow::create();
    bool initialized = false;
    int num_name = 1;
    string flow_name;
    while (true) {
        const int64 start = getTickCount();

        //build mats for the first frame
        if (!initialized) {
            bool success = video_stream->nextFrame(capture_frame);
            if (!success) break; // read frames until end
            capture_image.create(capture_frame.size(), CV_8UC4);
            capture_gray.create(capture_frame.size(), CV_8UC1);

            prev_image.create(capture_frame.size(), CV_8UC4);
            prev_gray.create(capture_frame.size(), CV_8UC1);

            capture_frame.copyTo(prev_image);
            cuda::cvtColor(prev_image, prev_gray, CV_BGRA2GRAY);
            initialized = true;
//            int c=prev_gray.channels();

            for (int s = 0; s < step; ++s) {
                video_stream->nextFrame(capture_frame);
            }
        } else {
            capture_frame.copyTo(capture_image);
            cuda::cvtColor(capture_image, capture_gray, CV_BGRA2GRAY);
            prev_gray.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
            capture_gray.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
            alg_tvl1->calc(d_frame0f, d_frame1f, d_flow);

            for (int s = 0; s < step - 1; ++s) {
                if (!video_stream->nextFrame(capture_frame))
                    break;
            }
            if (!video_stream->nextFrame(capture_frame))
                break;
            num_name += step;
            std::stringstream ss;
            ss << std::setw(5) << std::setfill('0') << num_name;
            std::string s = ss.str();
            flow_name = flow_dir + "/" + s + ".jpg";
            cuda::split(d_flow, planes);

            //get back flow map
            planes[0].download(flow_x);
            planes[1].download(flow_y);
            drawOpticalFlow(flow_x, flow_y, out, 10);

            if (!imwrite(flow_name, out))
                cout << "wrong save";

            std::swap(prev_gray, capture_gray);
            std::swap(prev_image, capture_image);
        }
        const double timeSec = (getTickCount() - start) / getTickFrequency();
//        cout<<flow_name<<" : "<<timeSec<<" sec"<<endl;
    }
}
