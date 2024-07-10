import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dataset.nyudepthv2 import nyudepthv2
from depth_anything.dataset.imagepath import imagepath
from depth_anything.util.logging import display_result
import depth_anything.util.metrics as metrics
from metric_depth.zoedepth.data.data_mono import DepthDataLoader
from metric_depth.zoedepth.utils.config import get_config

order = "--encoder vits --dataset imagepath --img-path F:\\Datasets\\NYUv2\\nyu_depth_v2\\official_splits\\test\\bathroom\\rgb_00045.jpg --outdir .\/res --crop_h 450 --crop_w 450"

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagepath', choices=['nyu', 'kitti', 'imagepath'])
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--crop_h', type=int, default=518)
    parser.add_argument('--crop_w', type=int, default=518)
    parser.add_argument('--max_depth_eval', type=float, default=10.0)
    parser.add_argument('--min_depth_eval', type=float, default=1e-3)
    parser.add_argument('--do_kb_crop', type=int, default=1)
    parser.add_argument("-m", "--model", type=str, default="zoedepth")
    
    args = parser.parse_args(order.split())
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('./checkpoints/depth_anything_vitb14', local_files_only=True).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    # config = get_config(args.model, "train", args.dataset)
    # dataset
    if args.dataset == 'nyu':
        val_dataset = DepthDataLoader(config, "online_eval").data
    elif args.dataset == 'imagepath':
        val_dataset = imagepath(args.img_path)
    else:
        raise NameError('No such dataset')

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    os.makedirs(args.outdir, exist_ok=True)

    # metric init
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    
    for batch in tqdm(val_loader):
        raw_image = batch['raw_image'].squeeze().cpu().numpy()
        image = batch['image'].to(DEVICE)
        # depth_gt = batch['depth'].to(DEVICE)
        filename = batch['filename'][0]

        h, w = raw_image.shape[0: 2]
        
        with torch.no_grad():
            depth = depth_anything(image)

        image_metric = (depth / depth.max() * 10.0).squeeze()
        depth_gt = (depth_gt / depth_gt.max() * 10.0).squeeze()

        # metric cal
        pred_crop, gt_crop = metrics.cropping_img(args, image_metric, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)

        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
        combined_results = cv2.hconcat([raw_image, split_region, depth_color])
        
        caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
        captions = ['Raw image', 'Depth Anything']
        segment_width = w + margin_width
        for i, caption in enumerate(captions):
            # Calculate text size
            text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

            # Calculate x-coordinate to center the text
            text_x = int((segment_width * i) + (w - text_size[0]) / 2)

            # Add text caption
            cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
        
        final_result = cv2.vconcat([caption_space, combined_results])
        
        filename = os.path.basename(filename)
        cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / len(val_loader)

    result_lines = display_result(result_metrics)
    print(result_lines)
