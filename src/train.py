"""Training script for road segmentation."""
import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import RoadDataset
from .model import build_unet
from .utils import dice_coef_tensor, iou_tensor
import segmentation_models_pytorch as smp

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--encoder', type=str, default='resnet34')
    return parser.parse_args()

def train(args):
    set_seed()
    device = args.device

    train_img_dir = os.path.join(args.data_dir, 'train')
    train_mask_dir = os.path.join(args.data_dir, 'train_labels')
    val_img_dir = os.path.join(args.data_dir, 'val')
    val_mask_dir = os.path.join(args.data_dir, 'val_labels')

    train_dataset = RoadDataset(train_img_dir, train_mask_dir, img_size=args.img_size, augment=True)
    val_dataset = RoadDataset(val_img_dir, val_mask_dir, img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_unet(encoder_name=args.encoder, encoder_weights='imagenet', in_channels=3, classes=1).to(device)

    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs(args.save_dir, exist_ok=True)

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    train_ious, val_ious = [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Train E{epoch+1}/{args.epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_coef_tensor(outputs.detach(), masks)
            running_iou += iou_tensor(outputs.detach(), masks)

        train_loss = running_loss / len(train_loader)
        train_dice = running_dice / len(train_loader)
        train_iou = running_iou / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_coef_tensor(outputs, masks)
                val_iou += iou_tensor(outputs, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)

if __name__ == '__main__':
    args = parse_args()
    train(args)
