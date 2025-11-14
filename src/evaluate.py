"""Evaluation and visualization script."""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .data import RoadDataset
from .model import build_unet
from .utils import dice_coef_tensor, iou_tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def visualize_batch(imgs, masks, preds, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    imgs = imgs.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    for i in range(min(4, imgs.shape[0])):
        img = imgs[i].transpose(1,2,0)
        img = img * np.array(std)[None,None,:] + np.array(mean)[None,None,:]
        img = np.clip(img, 0, 1)
        gt = masks[i].squeeze(0)
        pred = (preds[i].squeeze(0) > 0.5).astype(np.uint8)

        plt.figure(figsize=(9,3))
        plt.subplot(1,3,1); plt.imshow(img); plt.title("Image"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(gt, cmap='gray'); plt.title("GT"); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(pred, cmap='gray'); plt.title("Pred"); plt.axis('off')
        plt.show()

def evaluate(args):
    device = args.device
    test_img_dir = os.path.join(args.data_dir, 'test')
    test_mask_dir = os.path.join(args.data_dir, 'test_labels')

    test_dataset = RoadDataset(test_img_dir, test_mask_dir, img_size=args.img_size, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_unet().to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            test_dice += dice_coef_tensor(outputs, masks)
            test_iou += iou_tensor(outputs, masks)

    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    test_iou /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")

    # show a batch
    imgs, masks = next(iter(test_loader))
    with torch.no_grad():
        outputs = torch.sigmoid(model(imgs.to(device))).cpu()
    visualize_batch(imgs, masks, outputs)

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
