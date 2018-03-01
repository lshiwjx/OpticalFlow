#include "list_file.h"
#include "cal_optical_flow.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, const char* argv[])
{
    string filename1, filename2;
    Ptr<cuda::FarnebackOpticalFlow> tvl1 = cuda::FarnebackOpticalFlow::create();
    Ptr<cv::FarnebackOpticalFlow> tvl2 = cv::FarnebackOpticalFlow::create();
    string root = "/home/lshi/Database/Jester/20bn-jester-v1/";
    vector<string> dirs = getFiles(root);
    for (int i = 0; i < dirs.size(); ++i) {
        vector<string> jpgs = getFiles(root + dirs[i]);
        for (int j = 0; j < jpgs.size(); ++j) {
            if(j!=jpgs.size()-1){
                filename1=root + dirs[i]+"/"+jpgs[j];
                filename2=root + dirs[i]+"/"+jpgs[j+1];
                Mat frame0 = imread(filename1, IMREAD_GRAYSCALE);
                Mat frame1 = imread(filename2, IMREAD_GRAYSCALE);

                if (frame0.empty())
                {
                    cerr << "Can't open image ["  << filename1 << "]" << endl;
                    continue;
                }
                if (frame1.empty())
                {
                    cerr << "Can't open image ["  << filename2 << "]" << endl;
                    continue;
                }

                if (frame1.size() != frame0.size())
                {
                    cerr << "Images should be of equal sizes" << endl;
                    continue;
                }

                GpuMat d_frame0(frame0);
                GpuMat d_frame1(frame1);

                GpuMat d_flow(frame0.size(), CV_32FC2);

                const int64 start = getTickCount();
                tvl1->calc(d_frame0, d_frame1, d_flow);
                cv::Mat cpu_flow;
                d_flow.download(cpu_flow);
                const double timeSec = (getTickCount() - start) / getTickFrequency();
                cout << "TVL1 : " << timeSec << " sec" << endl;

                Mat flow(frame0.size(), CV_32FC2);
                const int64 start1 = getTickCount();
                tvl2->calc(frame0, frame1, flow);
                const double timeSec1 = (getTickCount() - start1) / getTickFrequency();
                cout << "TVL2 : " << timeSec1 << " sec" << endl;

//                cv::optflow::writeOpticalFlow("tvl1.flo", cpu_flow);
//                Mat show = cv::optflow::readOpticalFlow("tvl1.flo");

                showCpuFlow("TVL1", cpu_flow);
                showCpuFlow("TVL2", flow);
                waitKey(1);
            }

        }
    }

/*    Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
    Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();

    {
        GpuMat d_frame0f;
        GpuMat d_frame1f;

        d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
        d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

        const int64 start = getTickCount();

        brox->calc(d_frame0f, d_frame1f, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Brox : " << timeSec << " sec" << endl;

        showFlow("Brox", d_flow);
    }

    {
        const int64 start = getTickCount();

        lk->calc(d_frame0, d_frame1, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "LK : " << timeSec << " sec" << endl;

        showFlow("LK", d_flow);
    }

    {
        const int64 start = getTickCount();

        farn->calc(d_frame0, d_frame1, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Farn : " << timeSec << " sec" << endl;

        showFlow("Farn", d_flow);
    }*/

    return 0;
}