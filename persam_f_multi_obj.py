import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor



def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam_f')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')

    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--train_epoch_outside', type=int, default=1)
    parser.add_argument('--train_epoch_inside', type=int, default=200)
    parser.add_argument('--log_epoch', type=int, default=200)
    parser.add_argument('--training_percentage', type=float, default=0.5)
    
    parser.add_argument('--max_objects', type=int, default=10)
    parser.add_argument('--iou_threshold', type=float, default=0.8)
    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'
    output_path = './outputs/' + args.outdir

    

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    for obj_name in os.listdir(images_path):
        persam_f(args, obj_name, images_path, masks_path, output_path)


def persam_f(args, obj_name, images_path, masks_path, output_path):
    print("======> Load SAM" )
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    
    
    for name, param in sam.named_parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)

    print("\n------------> Segment " + obj_name)
    for i in tqdm(range(args.train_epoch_outside)):
        output_path = os.path.join(output_path, obj_name)
        os.makedirs(output_path, exist_ok=True)
        training_size = int(len(os.listdir(os.path.join(images_path, obj_name)))  * args.training_percentage)
        for ref_idx in range(training_size):
            # Path preparation
            ref_image_path = os.path.join(images_path, obj_name, '{:02}.jpg'.format(ref_idx))
            ref_mask_path = os.path.join(masks_path, obj_name, '{:02}.png'.format(ref_idx))
            test_images_path = os.path.join(images_path, obj_name)

            # Load images and masks
            ref_image = cv2.imread(ref_image_path)
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

            ref_mask = cv2.imread(ref_mask_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

            gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
            gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()
            
            # print("======> Obtain Self Location Prior" )
            # Image features encoding
            ref_mask = predictor.set_image(ref_image, ref_mask)
            ref_feat = predictor.features.squeeze().permute(1, 2, 0)

            ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
            ref_mask = ref_mask.squeeze()[0]

            # Target feature extraction
            target_feat = ref_feat[ref_mask > 0]
            target_feat_mean = target_feat.mean(0)
            target_feat_max = torch.max(target_feat, dim=0)[0]
            target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

            # Cosine similarity
            h, w, C = ref_feat.shape
            target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
            ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
            ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
            sim = target_feat @ ref_feat

            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()

            # Positive location prior
            topk_xy, topk_label = point_selection(sim, topk=1)


            # print('======> Start Training')
            # Learnable mask weights
            mask_weights = Mask_Weights().cuda()
            mask_weights.train()
            
            optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

            # Run the decoder
            masks, scores, logits, logits_high = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=True)
            original_logits_high = original_logits_high.flatten(1)

            for train_idx in range(args.train_epoch):
                # Weighted sum three-scale masks
                weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
                logits_high = original_logits_high * weights
                logits_high = logits_high.sum(0).unsqueeze(0)

                dice_loss = calculate_dice_loss(logits_high, gt_mask)
                focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
                loss = dice_loss + focal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            # print('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch_inside))
            current_lr = scheduler.get_last_lr()[0]


            mask_weights.eval()
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            weights_np = weights.detach().cpu().numpy()
            # print('======> Mask weights:\n', weights_np)
        print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))

    print('======> Start Testing')
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):

        # Load test image
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image_original = cv2.imread(test_image_path)
        test_image_original = cv2.cvtColor(test_image_original, cv2.COLOR_BGR2RGB)
        
        history_masks = []
        plt.figure(figsize=(10, 10))
        for i in tqdm(range(args.max_objects)):
            # Image feature encoding
            predictor.set_image(test_image)
            test_feat = predictor.features.squeeze()

            # Cosine similarity
            C, h, w = test_feat.shape
            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)
            sim = target_feat @ test_feat

            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()

            # Positive location prior
            topk_xy, topk_label = point_selection(sim, topk=1)

            # First-step prediction
            masks, scores, logits, logits_high = predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        multimask_output=True)

            # Weighted sum three-scale masks
            logits_high = logits_high * weights.unsqueeze(-1)
            logit_high = logits_high.sum(0)
            mask = (logit_high > 0).detach().cpu().numpy()

            logits = logits * weights_np[..., None]
            logit = logits.sum(0)

            # Cascaded Post-refinement-1
            y, x = np.nonzero(mask)
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :],
                mask_input=logit[None, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)

            # Cascaded Post-refinement-2
            y, x = np.nonzero(masks[best_idx])
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :],
                mask_input=logits[best_idx: best_idx + 1, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)


            final_mask = masks[best_idx]
            mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_colors[final_mask, :] = np.array([[0, 0, 128]])

            mask_bool = mask_colors.sum(axis=2) == 128
            test_image[mask_bool] = 0
            iou_over_threshold = False
            for h_mask in history_masks:
                if calculate_iou(h_mask, mask_colors) >= args.iou_threshold:
                    iou_over_threshold = True
                    break
            if iou_over_threshold:
                break
            show_mask(masks[best_idx], plt.gca())
            show_points(topk_xy, topk_label, plt.gca())
            history_masks.append(mask_colors)
        # Save masks
        
        plt.imshow(test_image_original)
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}_objects:{len(history_masks)}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')

        mask_output_path = os.path.join(output_path, test_idx + '.png')
        cv2.imwrite(mask_output_path, mask_colors)



class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    
    return topk_xy, topk_label


def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) score between two masks.

    Args:
        mask1: The first mask as a n*m*3 matrix with the mask parts being 128.
        mask2: The second mask as a n*m*3 matrix with the mask parts being 128.

    Returns:
        iou: The IoU score between the two masks.
    """

    mask1 = mask1.sum(axis=2)
    mask2 = mask2.sum(axis=2)

    mask1 = np.where(mask1 == 128, 1, 0)
    mask2 = np.where(mask2 == 128, 1, 0)
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    iou = intersection / union
    return iou

if __name__ == '__main__':
    main()
