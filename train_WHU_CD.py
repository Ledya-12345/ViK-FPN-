import os
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from model_ViK_FPN_CD import ViK_CD
from losses import BCEDiceLoss
from metric import Evaluator
from seed_utils import seed_everything, worker_init_fn
from config_utils import load_config, apply_config_to_args

# ======================================================
# Dataset (DIRECT PATH)
# ======================================================
class CDataset(Dataset):
    def __init__(self, root, augment=False):
        """
        root/
          ├── A/
          ├── B/
          └── label/

        augment: apply random horizontal/vertical flip and random rotation
        (Table 3: "Data Augmentation: Random Flip, Rotation"). Defaults to
        False so existing callers (validation / test) are unaffected unless
        explicitly opted in for training.
        """
        self.root = root
        self.augment = augment
        self.imgs1 = sorted(os.listdir(os.path.join(root, "A")))
        self.imgs2 = sorted(os.listdir(os.path.join(root, "B")))
        self.labels = sorted(os.listdir(os.path.join(root, "label")))

    def __len__(self):
        return len(self.labels)

    def _augment(self, img1, img2, mask):
        
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            mask = cv2.flip(mask, 1)
     
        if random.random() < 0.5:
            img1 = cv2.flip(img1, 0)
            img2 = cv2.flip(img2, 0)
            mask = cv2.flip(mask, 0)
       
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img1 = np.rot90(img1, k)
            img2 = np.rot90(img2, k)
            mask = np.rot90(mask, k)
        return np.ascontiguousarray(img1), np.ascontiguousarray(img2), np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        img1 = cv2.imread(os.path.join(self.root, "A", self.imgs1[idx]))
        img2 = cv2.imread(os.path.join(self.root, "B", self.imgs2[idx]))
        mask = cv2.imread(os.path.join(self.root, "label", self.labels[idx]), 0)

       
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img1, img2, mask = self._augment(img1, img2, mask)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask // 255).long()

        return {
            "img1": img1,
            "img2": img2,
            "gt_semantic_seg": mask
        }

# ======================================================
# Lightning Model
# ======================================================
class Supervision_Train(pl.LightningModule):
    def __init__(self, train_loader=None, val_loader=None, config=None):
        super().__init__()

        
        num_classes = config.num_classes if config else 2
        
        self.net = ViK_CD(num_classes=num_classes)
       
        self.loss = BCEDiceLoss(class_weights=(0.2, 0.8), num_classes=num_classes)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.metrics_train = Evaluator(num_class=num_classes)
        self.metrics_val = Evaluator(num_class=num_classes)

    def forward(self, img1, img2):
        return self.net(img1, img2)

    # ------------------ TRAIN ------------------
    def training_step(self, batch, batch_idx):
        img1 = batch["img1"]
        img2 = batch["img2"]
        mask = batch["gt_semantic_seg"]

        pred = self(img1, img2)
        loss = self.loss(pred, mask)

        pred_mask = torch.softmax(pred, dim=1).argmax(dim=1)

        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(
                mask[i].cpu().numpy(),
                pred_mask[i].cpu().numpy()
            )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())

        print(f"[TRAIN] mIoU={mIoU:.4f}, F1={F1:.4f}, OA={OA:.4f}")

        self.log_dict({
            "train_mIoU": mIoU,
            "train_F1": F1,
            "train_OA": OA
        }, prog_bar=True)

        self.metrics_train.reset()

    # ------------------ VAL ------------------
    def validation_step(self, batch, batch_idx):
        img1 = batch["img1"]
        img2 = batch["img2"]
        mask = batch["gt_semantic_seg"]

        pred = self(img1, img2)
        loss = self.loss(pred, mask)

        pred_mask = torch.softmax(pred, dim=1).argmax(dim=1)

        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(
                mask[i].cpu().numpy(),
                pred_mask[i].cpu().numpy()
            )

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())

        print(f"[VAL] mIoU={mIoU:.4f}, F1={F1:.4f}, OA={OA:.4f}")

        self.log_dict({
            "val_mIoU": mIoU,
            "val_F1": F1,
            "val_OA": OA
        }, prog_bar=True)

        self.metrics_val.reset()

    # ------------------ OPT ------------------
    def configure_optimizers(self):
        
        optimizer = optim.Adam(
            self.net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
        )
        try:
            max_epochs = self.trainer.max_epochs
        except RuntimeError:
            max_epochs = 100  
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=max_epochs, power=0.9
        )
       
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="dataset/WHU_CD/train")
    parser.add_argument("--val_root", type=str, default="dataset/WHU_CD/val")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)  # Table 3: Batch Size = 8
    parser.add_argument("--seed", type=int, default=42)       # Table 3: Random seed = 42
    parser.add_argument("--config", type=str, default=None,
                         help="Optional path to a YAML config (e.g. configs/whu_cd.yaml) "
                              "whose values override the CLI defaults above.")
    args = parser.parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
        flat = {
            "train_root": cfg.get("dataset", {}).get("train_root", args.train_root),
            "val_root": cfg.get("dataset", {}).get("val_root", args.val_root),
            "epochs": cfg.get("training", {}).get("epochs", args.epochs),
            "batch_size": cfg.get("training", {}).get("batch_size", args.batch_size),
            "seed": cfg.get("training", {}).get("seed", args.seed),
        }
        args = apply_config_to_args(args, flat)

    seed_everything(args.seed)

    train_set = CDataset(args.train_root, augment=True)   
    val_set = CDataset(args.val_root, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=lambda wid: worker_init_fn(wid, args.seed),
    )

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = Supervision_Train(train_loader, val_loader)

    checkpoint = ModelCheckpoint(
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="unetkan_cd-{epoch:02d}-{val_mIoU:.4f}"
    )

    logger = CSVLogger("lightning_logs", name="WHU_CD")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint],
        logger=logger,
  
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()
