import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process
from tqdm import tqdm

tool_path = '../cflow/calOpticalFlowFromImg'
video_tool_path = '../cflow/calOpticalFlowFromVideo'
step = 2
device = [0,1,2,3,4,6,7]


def tvl_flow(img_paths, flow_paths):
    l = len(img_paths)
    for index, (img_path, flow_path) in enumerate(zip(img_paths, flow_paths)):
        num_gpu = len(device)
        current = current_process()
        dev_id = (int(current._identity[0]) - 1) % num_gpu
        dev_id = device[dev_id]
        print(index, '/', l, '/', dev_id)
        cmd = tool_path + ' -i={} -f={} -s={} -d={}'.format(quote(img_path), quote(flow_path), int(step), int(dev_id))

        os.system(cmd)


def tvl_flow_video(video_paths, flow_paths):
    l = len(video_paths)
    for index, (video_path, flow_path) in tqdm(enumerate(zip(video_paths, flow_paths))):
        num_gpu = len(device)
        current = current_process()
        dev_id = (int(current._identity[0]) - 1) % num_gpu
        dev_id = device[dev_id]

        print(index, '/', l, '/', dev_id)
        cmd = video_tool_path + ' -i={} -f={} -s={} -d={}'.format(quote(video_path), quote(flow_path), int(step),
                                                                  int(dev_id))

        os.system(cmd)


def ucf():
    root = '/share/UCF-101/img/'
    target = '/home/lshi/Database/UCF101Flow/UCF101FlowTVL'

    img_paths = []
    flow_paths = []
    clip_dirs = sorted(os.listdir(root))
    for clip_dir in clip_dirs:
        img_dirs = sorted(os.listdir(os.path.join(root, clip_dir)))
        for img_dir in img_dirs:
            img_path = os.path.join(root, clip_dir, img_dir)
            flow_path = os.path.join(target, clip_dir, img_dir)
            l_img = len(os.listdir(img_path)) // 2
            l_flow = len(os.listdir(flow_path))
            if not os.path.isdir(flow_path):
                os.makedirs(flow_path)
            if not ((l_img == l_flow) or (l_img - 1 == l_flow)):
                img_paths.append(img_path)
                flow_paths.append(flow_path)

    # steps = [step for _ in range(len(img_paths))]
    # devices = [device for _ in range(len(img_paths))]
    # img_paths = img_paths[-100:]
    # flow_paths = flow_paths[-100:]
    num_workers = 2
    inter = len(img_paths) // num_workers
    img_paths_mul = [img_paths[i * inter:(i + 1) * inter] for i in range(num_workers)]
    img_paths_mul[-1] += (img_paths[inter * num_workers:])
    flow_paths_mul = [flow_paths[i * inter:(i + 1) * inter] for i in range(num_workers)]
    flow_paths_mul[-1] += (flow_paths[inter * num_workers:])
    pool = Pool(num_workers)
    pool.starmap(tvl_flow, zip(img_paths_mul, flow_paths_mul))


def ucf_video():
    root = '/home/lshi/Database/UCF101/img/'
    target = '/home/lshi/Database/UCF101Flow/UCF101FlowBrox'
    video_root = '/home/lshi/Database/UCF101/video'
    # video_target = '/home/lshi/Database/UCF101Flow/UCF101FlowTVLVideo'

    video_paths = []
    flow_paths = []
    clip_dirs = sorted(os.listdir(root))
    for clip_dir in tqdm(clip_dirs):
        img_dirs = sorted(os.listdir(os.path.join(root, clip_dir)))
        for img_dir in img_dirs:
            # img_path = os.path.join(root, clip_dir, img_dir)
            flow_path = os.path.join(target, clip_dir, img_dir)
            if not os.path.isdir(flow_path):
                os.makedirs(flow_path)
            # l_img = len(os.listdir(img_path)) // 2
            # l_flow = len(os.listdir(flow_path))
            # if not l_flow >= 50:
            video_paths.append(os.path.join(video_root, clip_dir, img_dir + '.avi'))
            flow_paths.append(os.path.join(target, clip_dir, img_dir))

    num_workers = 7
    inter = len(video_paths) // num_workers
    video_paths_mul = [video_paths[i * inter:(i + 1) * inter] for i in range(num_workers)]
    video_paths_mul[-1] += (video_paths[inter * num_workers:])
    flow_paths_mul = [flow_paths[i * inter:(i + 1) * inter] for i in range(num_workers)]
    flow_paths_mul[-1] += (flow_paths[inter * num_workers:])
    pool = Pool(num_workers)
    pool.starmap(tvl_flow_video, zip(video_paths_mul, flow_paths_mul))


def jester():
    root = '/opt/Jester/20bn-jester-v1/'
    target = '/home/lshi/Database/JesterFlow/TVL'

    img_paths = []
    flow_paths = []
    img_dirs = sorted(os.listdir(root))
    for img_dir in tqdm(img_dirs):
        img_path = os.path.join(root, img_dir)
        flow_path = os.path.join(target,img_dir)
        if not os.path.isdir(flow_path):
            os.makedirs(flow_path)
        if 0 == len(os.listdir(flow_path)):
            img_paths.append(img_path)
            flow_paths.append(flow_path)
    print(len(img_paths))
    # steps = [step for _ in range(len(img_paths))]
    # devices = [device for _ in range(len(img_paths))]
    num_workers = 7
    inter = len(img_paths) // num_workers
    img_paths_mul = [img_paths[i * inter:(i + 1) * inter] for i in range(num_workers)]
    img_paths_mul[-1] += (img_paths[inter * num_workers:])
    flow_paths_mul = [flow_paths[i * inter:(i + 1) * inter] for i in range(num_workers)]
    flow_paths_mul[-1] += (flow_paths[inter * num_workers:])
    pool = Pool(num_workers)
    pool.starmap(tvl_flow, zip(img_paths_mul, flow_paths_mul))


