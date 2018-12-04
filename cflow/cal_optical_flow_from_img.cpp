#include "list_file.h"
#include "cal_optical_flow.h"
#include "thread"
#include "mkdir.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;


class thread_guard {
    thread &t;
public :
    explicit thread_guard(thread &_t) :
            t(_t) {}

    ~thread_guard() {
        if (t.joinable())
            t.join();
    }

    thread_guard(const thread_guard &) = delete;

    thread_guard &operator=(const thread_guard &) = delete;
};

int main(int argc, const char *argv[]) {

    const char *keys =
            {
                    "{ i img_dir  | /home/lshi/Database/UCF101/img/ | dir of img }"
                            "{ f flow_dir  | /home/lshi/Database/UCF101FlowTest/ | dir of flow }"
                            "{ t num_dirs | 2 |  }"
                            "{ d dev | 0,2,3,4,5,6,7 | gpu id }"
                            "{ n num_gpu | 4  | }"
                            "{ w num_worker | 4  | }"
                            "{ s step  | 2 | step for frame sampling}"
            };
//
    CommandLineParser cmd(argc, argv, keys);
    string root = cmd.get<string>("img_dir");
    string flow_root = cmd.get<string>("flow_dir");
    int num_dirs = cmd.get<int>("num_dirs");
    int num_w = cmd.get<int>("num_worker");
    int num_g = cmd.get<int>("num_gpu");
    int step = cmd.get<int>("step");
    string dev = cmd.get<string>("dev");

    cout << "step: " << step << "num_dev: " << num_w << "num_dirs: " << num_dirs << "root: " << root << "flow_root: "
         << flow_root << endl;
    setenv("CUDA_VISIBLE_DEVICES", dev.c_str(), 1);
    cout << getenv("CUDA_VISIBLE_DEVICES") << endl;
//    compute_and_save_dir_flow(img_dir, flow_dir, step, device_id);

    vector<string> dirs_list, flow_list;

//    string root = "/home/lshi/Database/UCF101/img/";
//    string flow_root = "/home/lshi/Database/UCF101FlowTest/";
    if (num_dirs == 2) {
        vector<string> cls = getFiles(root);
        for (int i = 0; i < cls.size(); ++i) {
            string video_path = root + '/' + cls[i];
            string video_path_flow = flow_root + '/' + cls[i];
            vector<string> videos = getFiles(video_path);
            for (int j = 0; j < videos.size(); ++j) {
                string img_path = video_path + '/' + videos[j];
                string img_path_flow = video_path_flow + '/' + videos[j];
                dirs_list.push_back(img_path);
                flow_list.push_back(img_path_flow);
                if ((access(img_path_flow.c_str(), 0)) == -1) {
                    if (makePath(img_path_flow.c_str()))
                        cout << "make dir: " << img_path_flow << endl;
                    else
                        cout << "make dir failed: " << img_path_flow << endl;
                }

            }
        }
    } else {
        vector<string> videos = getFiles(root);
        for (int j = 0; j < videos.size(); ++j) {
            string img_path = root + '/' + videos[j];
            string img_path_flow = flow_root + '/' + videos[j];
            dirs_list.push_back(img_path);
            flow_list.push_back(img_path_flow);
            if ((access(img_path_flow.c_str(), 0)) == -1) {
                if (makePath(img_path_flow.c_str()))
                    cout << "make dir: " << img_path_flow << endl;
                else
                    cout << "make dir failed: " << img_path_flow << endl;
            }
        }
    }

    cout << dirs_list.size() << " num_list " << flow_list.size() << endl;
    thread t[num_w];
    ulong inter = dirs_list.size() / num_w;
    for (int k = 0; k < num_w; ++k) {
        vector<string>::const_iterator first = dirs_list.begin() + k * inter;
        vector<string>::const_iterator last = dirs_list.begin() + (k + 1) * inter;
        if (k == num_w - 1)
            last = dirs_list.end();
        vector<string> imgs(first, last);

        vector<string>::const_iterator firstf = flow_list.begin() + k * inter;
        vector<string>::const_iterator lastf = flow_list.begin() + (k + 1) * inter;
        if (k == num_w - 1)
            lastf = flow_list.end();
        vector<string> flows(firstf, lastf);
        int d = k % num_g;
        t[k] = thread(cal_flow_from_dir_list, imgs, flows, step, k);

        //thread_guard g(t[k]);
    }

    for (int l = 0; l < num_w; ++l) {
        thread_guard g(t[l]);
    }
    return 0;
}