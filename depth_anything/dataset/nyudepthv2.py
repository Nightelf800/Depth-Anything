# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
from torchvision.transforms import Compose

from depth_anything.dataset.base_dataset import BaseDataset
import json

from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class nyudepthv2(BaseDataset):
    def __init__(self, data_path, filenames_path='./depth_anything/dataset/filenames/',
                 is_train=True, crop_size=(518, 518), rescale_size=None):
        super().__init__(crop_size)

        self.rescale_size = rescale_size

        self.transform = Compose([
            Resize(
                width=crop_size[0],
                height=crop_size[1],
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'nyu_depth_v2')

        self.image_path_list = []
        self.depth_path_list = []

        txt_path = os.path.join(filenames_path, 'nyudepthv2')
        if is_train:
            txt_path += '/train_list.txt'
            self.data_path = self.data_path + '/sync'
        else:
            txt_path += '/test_list.txt'
            self.data_path = self.data_path + '/official_splits/test/'
 
        self.filenames_list = self.readTXT(txt_path) # debug
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        raw_image = cv2.imread(img_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        collection_trans = self.transform({'image': image, 'depth': depth})
        image = collection_trans['image']
        depth = collection_trans['depth']
        # image = image.unsqueeze()

        # print(image.shape, depth.shape, self.scale_size)
        depth = depth / 1000.0  # convert in meters

        return {'image': image, 'depth': depth, 'filename': filename, 'raw_image': raw_image}
