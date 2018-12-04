import torch
import vcap
import time
import cv2
import random
import numpy as np
# from torch.utils.cpp_extension import load
# vcap = load(
#     name='video_capture_resize_crop',
#     sources=['video_capture.cpp'],
#     extra_include_paths=[
#         '/home/lshi/Application/Anaconda/include',
#         "/home/lshi/Application/Anaconda/include/opencv/"
#     ],
#     extra_ldflags=[
#         '/home/lshi/Application/Anaconda/lib64/libopencv_cudacodec.so',
#         '/home/lshi/Application/Anaconda/lib64/libopencv_cudawarping.so'
#     ],
#     extra_cflags=['-O3'],
#     verbose=True)

final_shape = [1, 112, 112]
resize_min = 120
tmp_size = 1000
skip = 100
internal = 1

# file = "/home/lshi/Database/meitu/test_video/video/231125424.mp4"
file = "/home/lshi/Project/caffe2/interesting/interesting_torch/test/video/963193352.mp4"
tmp = torch.cuda.ByteTensor(final_shape[0], tmp_size, tmp_size, 4)
final = torch.cuda.ByteTensor(final_shape[0], final_shape[1], final_shape[1], 4)
# vcap.video_capture_resize_crop(file, tmp, final, resize_min, final_shape[1], final_shape[0], skip, internal, pw, ph)


for i in range(4):
    s = time.perf_counter()
    pw = random.randint(0, 2)
    ph = random.randint(0, 2)
    vcap.video_capture_resize_crop(file, tmp, final, resize_min, final_shape[1], final_shape[0], skip, internal, pw, ph)
    inputs = final.unsqueeze(0).float()
    torch.cuda.synchronize()
    print(time.perf_counter() - s)

clip = np.ndarray((final_shape[0], final_shape[1], final_shape[2], 3), dtype=np.float32)
for i in range(4):
    s = time.perf_counter()
    video = cv2.VideoCapture(file, cv2.CAP_FFMPEG)
    # video.set(cv2.CAP_PROP_POS_FRAMES, skip)
    for i in range(skip):
        ret = video.grab()
        assert ret==True
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    size = [height // width * resize_min, resize_min] if height > width else [resize_min, width // height * resize_min]

    gap = [size[i] - final_shape[i + 1] for i in range(2)]
    starts = [[0, 0], [0, gap[1] // 2], [0, gap[1]],
              [gap[0] // 2, 0], [gap[0] // 2, gap[1] // 2], [gap[0] // 2, gap[1]],
              [gap[0], 0], [gap[0], gap[1] // 2], [gap[0], gap[1]]]
    start = random.sample(starts, 1)[0]
    for index in range(final_shape[0]):
        for i in range(internal):
            ret, frame = video.read()
        if not ret:
            print(file, ' not read')
            for i in range(index, final_shape[0]):
                clip[i] = clip[index - 1]
            break
        if frame is None:
            for i in range(index, final_shape[0]):
                clip[i] = clip[index - 1]
            break
        frame = cv2.resize(frame, (size[1], size[0]))
        frame = frame[start[0]:start[0] + final_shape[1], start[1]:start[1] + final_shape[2], :]
        clip[index] = frame.astype(np.float32)
    inputs = torch.from_numpy(clip).cuda(non_blocking=True).unsqueeze(0)
    torch.cuda.synchronize()
    print(time.perf_counter() - s)
