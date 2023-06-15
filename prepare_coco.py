import os
import json
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse

def copy_file(src, dst):
    shutil.copy(src, dst)

def coco2mask(coco_file_path, img_save_path):
    print('coco_file_path:', coco_file_path)
    print('img_save_path:', img_save_path)
    with open(coco_file_path, 'r') as f:
        coco_json = json.load(f)

    class_mapper = dict()
    for category in coco_json['categories']:
        class_mapper[category['id']] = category['name']
        os.makedirs(os.path.join(img_save_path,category['name']), exist_ok=True)
        os.makedirs(os.path.join('data', 'Images', category['name']), exist_ok=True)
    for category in coco_json['categories']:
        print(category)
        idx = 0
        for image in coco_json['images']:
            height = image['height']
            width = image['width']
            mask = np.zeros((height, width), dtype=np.uint8)
            
            has_annots = False
            for annotation in coco_json['annotations']:
                if annotation['segmentation'] == []:
                    continue
                if annotation['category_id'] == category['id'] and annotation['image_id'] == image['id']:
                    has_annots = True
                    seg = annotation['segmentation']
                    seg = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    mask = cv2.fillPoly(mask, [seg], 128)
            if has_annots:
            
                mask_img = np.zeros((height, width, 3), dtype=np.uint8)
                mask_img[:, :, 0] = mask
                mask_img[:, :, 1] = mask
                mask_img[:, :, 2] = mask
                img_save_name = os.path.join(img_save_path, class_mapper[category['id']], "{:02}".format(idx) + '.png')
                mask_img[:,:,1:] = 0
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)  # 将参考图像从BGR颜色空间转为RGB颜色空间
                copy_file(os.path.join('auto-sam-data/',image['file_name']), 
                os.path.join('data', 'Images',class_mapper[category['id']], "{:02}".format(idx) + '.jpg')
                )
                
                cv2.imwrite(img_save_name, mask_img)
                idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_path', type=str, help='path to JSON file')
    parser.add_argument('output_dir', type=str, help='output directory')
    args = parser.parse_args()

    coco2mask(os.path.join(args.coco_path, 'result.json'), args.output_dir)

if __name__ == '__main__':
    main()
