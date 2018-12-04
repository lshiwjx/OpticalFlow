import cv2
import os
import time
import numpy as np
import utility.flow as fl

os.environ['DISPLAY'] = 'localhost:11.0'
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4,3,2,1,0'


def dense_flow_ucf(cls, root, target):
    deep_flow = cv2.optflow.createOptFlow_DeepFlow()
    start = time.time()
    cls_path = os.path.join(root, cls)
    clips = sorted(os.listdir(cls_path))
    for clip in clips:
        clip_path = os.path.join(cls_path, clip)
        imgs = sorted(os.listdir(clip_path))
        pre = None
        for index, img in enumerate(imgs):
            img_path = os.path.join(clip_path, img)
            if index == 0:
                pre = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                after = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                flow = deep_flow.calc(pre, after, None)
                save_path = os.path.join(target, cls, clip)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                    duration = time.time() - start
                    start = time.time()
                    print('save dir: ', save_path, ' time: ', duration)
                if not cv2.optflow.writeOpticalFlow(save_path + '/' + img[:-4] + '.flo', flow):
                    print('save failed: ', save_path)


def compare_flow():
    img1_path = "/home/lshi/Database/UCF101/img/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00001.jpg"
    img2_path = "/home/lshi/Database/UCF101/img/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00005.jpg"
    tv1 = cv2.optflow.createOptFlow_PCAFlow()
    df = cv2.optflow.op()
    dis = cv2.optflow.createOptFlow_DIS()
    far = cv2.optflow.createOptFlow_Farneback()
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    start = time.time()
    flow1 = df.calc(img1, img2, None)
    # fl.visualize_flow(flow)
    print(time.time() - start)

    start = time.time()
    flow2 = dis.calc(img1, img2, None)
    # fl.visualize_flow(flow)
    print(time.time() - start)

    start = time.time()
    flow3 = far.calc(img1, img2, None)
    # fl3.visualize_flow(flow)
    print(time.time() - start)

    start = time.time()
    flow4 = tv1.calc(img1, img2, None)
    # fl.visualize_flow(flow)
    print(time.time() - start)
    fl.visualize_flow(flow1)
    fl.visualize_flow(flow2)
    fl.visualize_flow(flow3)
    fl.visualize_flow(flow4)
    pass



def tvl_flow_ucf(p):
    cls, root, target = p
    tv1 = cv2.createOptFlow_DualTVL1()
    cls_path = os.path.join(root, cls)
    clips = sorted(os.listdir(cls_path))
    for clip in clips:
        clip_path = os.path.join(cls_path, clip)
        imgs = sorted(os.listdir(clip_path))
        pre = None
        for index, img in enumerate(imgs):
            img_path = os.path.join(clip_path, img)
            if index == 0:
                pre = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            elif index % 2 == 0:
                start = time.time()
                after = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                flow = tv1.calc(pre, after, None)
                save_path = os.path.join(target, cls, clip)
                duration = time.time() - start
                print('save dir: ', save_path, ' time: ', duration)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                flow_img = fl.flow_to_image(flow)
                if not cv2.imwrite(save_path + '/' + img[:-4] + '.jpg', flow_img):
                    print('save failed: ', save_path)


def ucf():
    root = '/home/lshi/Database/UCF101/'
    target = '/home/lshi/Database/UCF101FlowJPGTV1/'
    classes = sorted(os.listdir(root))
    for cls in classes:
        tvl_flow_ucf((cls, root, target))
        # roots = [root for _ in range(len(classes))]
        # targets = [target for _ in range(len(classes))]
        # pool = Pool(2)
        # p = zip(classes, roots, targets)
        # pool.map(tvl_flow_ucf, p)


compare_flow()
