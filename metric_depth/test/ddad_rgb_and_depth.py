import numpy as np
import os
from PIL import Image


if __name__ == '__main__':
    ddad_path = '/home/ylc/datasets/DDAD/ddad_train_val'

    id = '15621787674931530'
    depth_path = os.path.join(ddad_path, '000000/depth/lidar/CAMERA_01', id + '.npz')


    depth = np.load(depth_path)['depth']  # meters
    # print(f'image: {image.shape}')
    print(f'depth_npy: {depth.shape}')
    # print('image_path.split: {}'.format(image_path.split('/')))
    # print(f'image.max: {image.max()}')
    # print(f'image.min: {image.min()}')
    print(f'depth.max: {depth.max()}')
    print(f'depth.min: {depth.min()}')

    depth = depth * 5
    print(f'depth.max: {depth.max()}')
    print(f'depth.min: {depth.min()}')

    # ori_img = Image.fromarray((image * 255).astype(np.uint8))
    depth_img = Image.fromarray(depth.astype(np.uint8))
    depth_RGB = depth_img.convert('RGB')
    # ori_RGB = ori_img.convert('RGB')
    if not os.path.exists('./imgs'):
        os.makedirs('./imgs')
    depth_RGB.save('./imgs/{}_5*depth.jpeg'.format(id))
    # ori_RGB.save('./imgs/{}.jpeg'.format(id))
    print('img save successfully')