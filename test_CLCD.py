import os
import random
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_CLCD import Supervision_Train, CDataset
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from metric import Evaluator
from utils import save_result
from seed_utils import seed_everything
from checkpoint_utils import find_best_checkpoint


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
  
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-o", "--output_path", type=Path, required=True, help="Path where to save resulting masks.")
    arg("--rgb", help="whether output rgb images", action='store_true')
    arg("--weights_path", type=str, default="lightning_logs/CLCD/version_2/checkpoints/", help="Path to the directory containing the checkpoint.")
    arg("--test_weights_name", type=str, default="unetkan_cd-epoch=29-val_mIoU=0.7091", help="Name of the checkpoint file without extension. Ignored if --auto_best is set.")
    arg("--auto_best", action='store_true', help="Automatically pick the checkpoint in --weights_path with the "
                                                   "highest val_mIoU instead of using --test_weights_name.")
    arg("--test_root", type=str, default="dataset/CLCD/test", help="Path to the test dataset.")
    return parser.parse_args()


class Config:
    def __init__(self, args):
        self.weights_path = args.weights_path
        self.test_weights_name = args.test_weights_name
        self.num_classes = 2
        self.classes = ['background', 'change']
        self.test_dataset = CDataset(args.test_root)


def main():
    seed_everything(42)
    args = get_args()
    config = Config(args)
    args.output_path.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Supervision_Train(config=config)  # create model without loading checkpoint
    model.to(device)
    model.eval()
    # dummy forward to build dynamic parameters
    dummy_img = torch.randn(1, 3, 256, 256, device=device)
    _ = model(dummy_img, dummy_img)
    # now load checkpoint
    if args.auto_best:
        best = find_best_checkpoint(config.weights_path, metric_name="val_mIoU", mode="max")
        if best is None:
            raise FileNotFoundError(f"No checkpoint matching 'val_mIoU=' found in {config.weights_path}")
        checkpoint_path = best
    else:
        checkpoint_path = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Either train a model first (see train_CLCD.py / README.md), download the "
            f"pretrained weights (see scripts/download_pretrained_weights.py), or pass "
            f"--auto_best to pick the best checkpoint automatically from --weights_path."
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        global_idx = 0
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img1'].to(device), input['img2'].to(device))

            image_ids = [f"result_{global_idx + i}" for i in range(len(input['img1']))]  # Assuming image_ids are not provided, use indices
            masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))
                save_result(input['img1'][i].cpu(), input['img2'][i].cpu(), masks_true[i].cpu(), predictions[i].cpu(), f"outputsCLCD/result_{global_idx + i}.png")
                global_idx += 1

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    precision_per_class = evaluator.Precision()
    recall_per_class = evaluator.Recall()
    for class_name, class_iou, class_f1, class_precision, class_recall in zip(config.classes, iou_per_class, f1_per_class, precision_per_class, recall_per_class):
        print('F1_{}:{}, IOU_{}:{}, Precision_{}:{}, Recall_{}:{}'.format(class_name, class_f1, class_name, class_iou, class_name, class_precision, class_name, class_recall))
    print('F1(mean):{}, mIOU:{}, OA:{}, Precision(mean):{}, Recall(mean):{}'.format(
        np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA,
        np.nanmean(precision_per_class), np.nanmean(recall_per_class)))

    # Sec. 4.3 / Tables 4-5: reported Precision, Recall, F1, and IoU correspond
    # to the change class only (index 1 of config.classes = ['background',
    # 'change']), not the mean over both classes printed above.
    change_idx = config.classes.index('change')
    print('--- Change-class metrics (as reported in Tables 4-5) ---')
    print('Recall:{:.4f}, Precision:{:.4f}, IoU:{:.4f}, F1:{:.4f}, OA:{:.4f}'.format(
        recall_per_class[change_idx], precision_per_class[change_idx],
        iou_per_class[change_idx], f1_per_class[change_idx], OA))
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))
if __name__ == "__main__":
    main()
