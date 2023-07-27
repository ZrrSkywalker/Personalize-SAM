import os
import errno
import numpy as np
from PIL import Image
import warnings
from davis2017.davis import DAVIS


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_semantic_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img


def generate_obj_proposals(davis_root, subset, num_proposals, save_path):
    dataset = DAVIS(davis_root, subset=subset, codalab=True)
    for seq in dataset.get_sequences():
        save_dir = os.path.join(save_path, seq)
        if os.path.exists(save_dir):
            continue
        all_gt_masks, all_masks_id = dataset.get_all_masks(seq, True)
        img_size = all_gt_masks.shape[2:]
        num_rows = int(np.ceil(np.sqrt(num_proposals)))
        proposals = np.zeros((num_proposals, len(all_masks_id), *img_size))
        height_slices = np.floor(np.arange(0, img_size[0] + 1, img_size[0]/num_rows)).astype(np.uint).tolist()
        width_slices = np.floor(np.arange(0, img_size[1] + 1, img_size[1]/num_rows)).astype(np.uint).tolist()
        ii = 0
        prev_h, prev_w = 0, 0
        for h in height_slices[1:]:
            for w in width_slices[1:]:
                proposals[ii, :, prev_h:h, prev_w:w] = 1
                prev_w = w
                ii += 1
                if ii == num_proposals:
                    break
            prev_h, prev_w = h, 0
            if ii == num_proposals:
                break

        os.makedirs(save_dir, exist_ok=True)
        for i, mask_id in enumerate(all_masks_id):
            mask = np.sum(proposals[:, i, ...] * np.arange(1, proposals.shape[0] + 1)[:, None, None], axis=0)
            save_mask(mask, os.path.join(save_dir, f'{mask_id}.png'))


def generate_random_permutation_gt_obj_proposals(davis_root, subset, save_path):
    dataset = DAVIS(davis_root, subset=subset, codalab=True)
    for seq in dataset.get_sequences():
        gt_masks, all_masks_id = dataset.get_all_masks(seq, True)
        obj_swap = np.random.permutation(np.arange(gt_masks.shape[0]))
        gt_masks = gt_masks[obj_swap, ...]
        save_dir = os.path.join(save_path, seq)
        os.makedirs(save_dir, exist_ok=True)
        for i, mask_id in enumerate(all_masks_id):
            mask = np.sum(gt_masks[:, i, ...] * np.arange(1, gt_masks.shape[0] + 1)[:, None, None], axis=0)
            save_mask(mask, os.path.join(save_dir, f'{mask_id}.png'))


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def save_mask(mask, img_path):
    if np.max(mask) > 255:
        raise ValueError('Maximum id pixel value is 255')
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.putpalette(color_map().flatten().tolist())
    mask_img.save(img_path)


def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


def list_files(dir, extension=".png"):
    return [os.path.splitext(file_)[0] for file_ in os.listdir(dir) if file_.endswith(extension)]


def force_symlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
        os.symlink(file1, file2)
