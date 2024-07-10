# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose

from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class imagepath(Dataset):
    # for test only
    def __init__(self, data_path, crop_size=(518, 518),):
        super().__init__()

        self.data_path = data_path

        self.transform = Compose([
        Resize(
            width=crop_size[0],
            height=crop_size[1],
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

        if os.path.isfile(data_path):
            if data_path.endswith('txt'):
                with open(data_path, 'r') as f:
                    self.filenames = f.read().splitlines()
            else:
                self.filenames = [data_path]
        else:
            self.filenames = os.listdir(data_path)
            self.filenames = [os.path.join(data_path, filename) for filename in self.filenames if
                         not filename.startswith('.')]
            self.filenames.sort()

        print("Dataset : Image Path")
        print("# of images: %d" % (len(self.filenames)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = self.filenames[idx]
        raw_image = cv2.imread(file)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = self.transform({'image': image})['image']
        # image = image.unsqueeze()

        filename = file.split('/')[-1]

        return {'image': image, 'filename': filename, 'raw_image': raw_image}
