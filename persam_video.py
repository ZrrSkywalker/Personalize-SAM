import argparse, os
from PIL import Image
from os import path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from per_segment_anything import SamPredictor, sam_model_registry
from davis2017.davis import DAVISTestDataset, all_to_onehot
from eval_video import eval_davis_result

def main(args):
    if args.eval:
        eval_davis_result(args.output_path, args.davis_path)
        return

    # Dataset
    print("Running on DAVIS", args.dataset_set)
    test_dataset = DAVISTestDataset(args.davis_path, imset=args.dataset_set + '/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    palette = Image.open(path.expanduser(os.path.join(args.davis_path, 'Annotations/480p/bike-packing/00000.png'))).getpalette()

    # Load SAM
    sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    predictor = SamPredictor(sam)

    # Start eval
    for iter, data in enumerate(test_loader):
        rgb = data['rgb'].cpu().numpy()
        msk = data['gt'][0].cpu().numpy()
        info = data['info']
        name = info['name'][0]
        os.makedirs(args.output_path, exist_ok=True)
        L = os.listdir(args.output_path)
        print("Processing Video", name, "....")
        if name in L:
            print("File", name, "exists in", args.output_path, ", skip...")
            continue
        num_obj = len(info['labels'][0])
   
        frame_num = rgb.shape[1]

        save_path = args.output_path + '/{}/'.format(name)
        os.makedirs(save_path, exist_ok=True)
        first_frame_image = rgb[0, 0] 
        first_frame_mask = msk[:, 0] * args.exp 
        
        fore_feat_list = []
        # Foreground features
        input_boxes = []
        for k in range(msk[:, 0].shape[0]):
            input_boxes.append(msk[:, 0][k])
        for obj in range(num_obj):
            print("Processing Object", obj)
            frame_image = first_frame_image

            obj_mask = first_frame_mask[obj].reshape(first_frame_mask.shape[1], first_frame_mask.shape[2], 1)
            obj_mask = np.concatenate((obj_mask, np.zeros((obj_mask.shape[0], obj_mask.shape[1], 2), dtype=obj_mask.dtype)), axis=2)
            obj_mask = predictor.set_image(frame_image, obj_mask)
            if obj == 0:
                img_feat1 = predictor.features.squeeze().permute(1, 2, 0)
            obj_mask = F.interpolate(obj_mask, size=img_feat1.shape[0:2], mode="bilinear")
            obj_mask = obj_mask.squeeze()[0]

            fore_feat = img_feat1[obj_mask > 0]

            if fore_feat.shape[0] == 0:
                fore_feat_list.append(fore_feat.mean(0))
                print("Find a small object in", name, "Object", obj)
                continue

            fore_feat_mean = fore_feat.mean(0)
            fore_feat_max = torch.max(fore_feat, dim=0)[0]
            fore_feat = (fore_feat_max / 2 + fore_feat_mean / 2).unsqueeze(0)
            fore_feat = fore_feat / fore_feat.norm(dim=-1, keepdim=True)
            fore_feat_list.append(fore_feat)

        for i in range (1, frame_num):
            current_img = rgb[0, i] 
            predictor.set_image(current_img)

            # pred masks
            test_feat = predictor.features.squeeze()
            C, htest, wtest = test_feat.shape

            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, htest * wtest)
            
            concat_mask = np.zeros((1, first_frame_mask.shape[1], first_frame_mask.shape[2]), dtype=np.uint8)
            for j in range(min(len(fore_feat_list), len(input_boxes))):
                # Cosine similarity
                fore_feat = fore_feat_list[j]
                sim = fore_feat @ test_feat  # 1, h*w
                sim = sim.reshape(1, 1, htest, wtest)
                sim = F.interpolate(sim, scale_factor=4, mode="bilinear")

                mask_sim = predictor.model.postprocess_masks(
                                sim,
                                input_size=predictor.input_size,
                                original_size=predictor.original_size).squeeze()

                # Top-k point selection
                w, h = mask_sim.shape

                topk_xy_i, topk_label_i = point_selection(mask_sim, topk=args.topk)
                topk_xy = topk_xy_i
                topk_label = topk_label_i

                if args.center:
                    topk_label = np.concatenate([topk_label, [1]], axis=0)
                
                if args.box_prompt:
                    center, input_box_ = get_box_prompt(input_boxes[j], args.threshold)
                    if args.center:
                        topk_xy = np.concatenate((topk_xy, center), axis=0)

                    masks, scores, logits, _ = predictor.predict(
                                point_coords=topk_xy,
                                point_labels=topk_label,
                                box=input_box_[None, :],
                                multimask_output=True)
                else:
                    masks, scores, logits, _ = predictor.predict(
                                point_coords=topk_xy,
                                point_labels=topk_label,
                                multimask_output=True)
                if args.large:
                    masks_ = masks 
                    mask_num = np.array([np.sum(masks_[0]), np.sum(masks_[1]), np.sum(masks_[2])], dtype=np.uint8)
                    ic_index = np.argmax(mask_num, axis=0).astype(np.uint8)
                else:
                    ic_index = 0

                masks, scores, logits, _ = predictor.predict(
                            point_coords=topk_xy,
                            point_labels=topk_label,
                            mask_input=logits[ic_index: ic_index + 1, :, :], 
                            multimask_output=True)
                ic_index = np.argmax(scores)

                # box refine
                y, x = np.nonzero(masks[ic_index])
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
                masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits[ic_index: ic_index + 1, :, :], 
                    multimask_output=True,
                    return_logits=True)

                ic_index = np.argmax(scores)

                concat_mask = np.concatenate((concat_mask, masks[ic_index].reshape(1, masks.shape[1], masks.shape[2])), axis=0)
                
            current_mask_pred = np.argmax(concat_mask, axis=0).astype(np.uint8)
            output = Image.fromarray(current_mask_pred)
            output.putpalette(palette)
            output.save(save_path + '{:05d}.png'.format(i))

            if args.box_prompt:
                cur_labels = np.unique(current_mask_pred)
                cur_labels = cur_labels[cur_labels!=0]
                input_boxes = all_to_onehot(current_mask_pred, cur_labels)
            
        print(f"Finish predict video: {name}")

    eval_davis_result(args.output_path, args.davis_path)

def get_box_prompt(img, threshold):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cmin = 0 if cmin - threshold <= 0 else cmin - threshold 
    rmin = 0 if rmin - threshold <= 0 else rmin - threshold
    cmax = img.shape[1] if cmax + threshold >= img.shape[1] else cmax + threshold
    rmax = img.shape[0] if rmax + threshold >= img.shape[0] else rmax + threshold
   
    return np.array([[(cmin + cmax) // 2, (rmin + rmax) // 2]]), np.array([cmin,rmin,cmax,rmax]) # x1,y1,x2,y2

def point_selection(mask_sim, topk=1):
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    return topk_xy, topk_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="output path", required=True)
    parser.add_argument('--davis_path', default='./DAVIS/2017')
    parser.add_argument("--dataset_set", type=str, help="2017", default='2017')
    parser.add_argument("--topk", type=int, help="choose topk points", default=2)
    parser.add_argument("--exp", type=int, help="expand mask value to", default=215)
    parser.add_argument("--threshold", type=int, help="the threshold for bounding box expansion", default=10)
    parser.add_argument("--eval", action="store_true", help="eval only")
    parser.add_argument("--box_prompt", action="store_true", help="whether use box prompt")
    parser.add_argument("--large", action="store_true", help="whether choose largest mask for prompting after stage 1")
    parser.add_argument("--center", action="store_true", help="whether prompt with center")
    parser.set_defaults(box_prompt=True)
    parser.set_defaults(large=True)
    parser.set_defaults(center=True)
    args = parser.parse_args()
    print(args)
    main(args)