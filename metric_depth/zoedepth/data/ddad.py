# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os
import time
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from VFDepth.dataset.data_util import mask_loader_scene


class ToTensor(object):
    def __init__(self, resize_shape):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalize = lambda x : x
        self.resize = transforms.Resize(resize_shape)

    def __call__(self, sample):
        image, depth, mask = sample['image'], sample['depth'], sample['mask']
        image = self.to_tensor(image)
        # image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "ddad", 'mask': mask}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class DDAD(Dataset):
    def __init__(self, data_dir_root, resize_shape):
        import glob

        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        
        # self.image_files = glob.glob(os.path.join(data_dir_root, '*.png'))
        # self.depth_files = [r.replace("_rgb.png", "_depth.npy")
        #                     for r in self.image_files]
        self.image_files, self.depth_files = [], []
        with open('./zoedepth/data/val.txt', 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            self.image_files.append(line.split(' ')[0])
            self.depth_files.append(line.split(' ')[1])

        self.mask_path = "./VFDepth/dataset/ddad_mask"
        mask_file_name = os.path.join(self.mask_path, 'mask_idx_dict.pkl')
        self.mask_idx_dict = pd.read_pickle(mask_file_name)
        self.mask_loader = mask_loader_scene

        self.transform = ToTensor(resize_shape)

    def __getitem__(self, idx):

        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.load(depth_path)['depth'] / 256.0  # meters
        # print(f'image: {image.shape}')
        # print(f'depth_npy: {depth.shape}')
        # print(f'depth_npy: {depth.dtype}')
        # print('image_path.split: {}'.format(image_path.split('/')))
        # print(f'image.max: {image.max()}')
        # print(f'image.min: {image.min()}')
        # print(f'depth.max: {depth.max()}')
        # print(f'depth.min: {depth.min()}')
        #
        #
        # ori_img = Image.fromarray((image * 255).astype(np.uint8))
        # depth_img = Image.fromarray(depth.astype(np.uint8))
        # depth_RGB = depth_img.convert('RGB')
        # ori_RGB = ori_img.convert('RGB')
        # if not os.path.exists('./results/img/{}/{}'.format(image_path.split('/')[-4], image_path.split('/')[-2])):
        #     os.makedirs('./results/img/{}/{}'.format(image_path.split('/')[-4], image_path.split('/')[-2]))
        # depth_RGB.save('./results/img/{}/{}/{}_depth.jpeg'.format(image_path.split('/')[-4], image_path.split('/')[-2], image_path.split('/')[-1].split('.')[-2]))
        # ori_RGB.save('./results/img/{}/{}/{}.jpeg'.format(image_path.split('/')[-4], image_path.split('/')[-2],
        #                                                           image_path.split('/')[-1].split('.')[-2]))
        # print('img save successfully')
        # exit()

        mask_idx = self.mask_idx_dict[int(image_path.split('/')[-4])]
        mask = self.mask_loader(self.mask_path, mask_idx, image_path.split('/')[-2])
        mask = np.asarray(mask, dtype=np.int32)

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth, mask=mask)
        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_ddad_loader(data_dir_root, resize_shape, batch_size=1, **kwargs):
    dataset = DDAD(data_dir_root, resize_shape)
    return DataLoader(dataset, batch_size, **kwargs)

def ddad_preprocess(data_dir_root, txt_path):
    rgb_file_num = 0
    depth_file_num = 0

    data_dir_root = os.path.join(data_dir_root, 'ddad_train_val')
    txt_content = ""

    for dir_index in os.listdir(data_dir_root):
        data_dir = os.path.join(data_dir_root, dir_index)
        if not os.path.isdir(data_dir):
            continue
        rgb_dir = os.path.join(data_dir, 'rgb')
        depth_dir = os.path.join(data_dir, 'depth', 'lidar')

        for camera_index in os.listdir(depth_dir):
            depth_cam_dir = os.path.join(depth_dir, camera_index)

            for file_index in os.listdir(depth_cam_dir):
                file_name, file_type = file_index.split('.')

                rgb_file = os.path.join(rgb_dir, camera_index, file_name + '.png')
                depth_file = os.path.join(depth_cam_dir, file_index)

                if not os.path.isfile(rgb_file):
                    print(f'\nrgb_file: {rgb_file}')
                    raise FileNotFoundError
                else:
                    rgb_file_num += 1
                if not os.path.isfile(depth_file):
                    print(f'\ndepth_file: {depth_file}')
                    raise FileNotFoundError
                else:
                    depth_file_num += 1

                txt_content += rgb_file + ' ' + depth_file + '\n'


                # print(f'rgb files Found: {rgb_file_num}, depth files Found: {depth_file_num}', end='\r')
    print(f'write into {txt_path}')

    with open(txt_path, "w") as f:
        f.write(txt_content)

    print(f'\nsave successfully')




if __name__ == '__main__':
    ddad_preprocess("/home/ylc/datasets/DDAD", "val.txt")
