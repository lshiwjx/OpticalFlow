import os

root = '/home/xxx/Database/UCF101'
root2 = '/home/xxx/Database/UCF101FlowTVL'
dirs = os.listdir(root)
for d in dirs:
    clips = os.listdir(os.path.join(root,d))
    for clip in clips:
        os.makedirs(os.path.join(root2, d, clip))