# --------------------------------------------------------
# PersonalizeSAM -- Personalize Segment Anything Model with One Shot
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from PIL import Image
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from torch.nn import functional as F

from show import *
from per_segment_anything import sam_model_registry, SamPredictor


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


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
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label


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


def inference(ic_image, ic_mask, image1, image2):
    # in context image and mask
    ic_image = np.array(ic_image.convert("RGB"))
    ic_mask = np.array(ic_mask.convert("RGB"))
    
    sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    # sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    predictor = SamPredictor(sam)
    
    # Image features encoding
    ref_mask = predictor.set_image(ic_image, ic_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]
    
    # Target feature extraction
    print("======> Obtain Location Prior" )
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)
    
    output_image = []
    
    for test_image in [image1, image2]:
        print("======> Testing Image" )
        test_image = np.array(test_image.convert("RGB"))
        
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
        
        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
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
        mask_colors[final_mask, :] = np.array([[128, 0, 0]])
        output_image.append(Image.fromarray((mask_colors * 0.6 + test_image * 0.4).astype('uint8'), 'RGB'))
    
    return output_image[0].resize((224, 224)), output_image[1].resize((224, 224))


def inference_scribble(image, image1, image2):
    # in context image and mask
    ic_image = image["image"]
    ic_mask = image["mask"]
    ic_image = np.array(ic_image.convert("RGB"))
    ic_mask = np.array(ic_mask.convert("RGB"))
    
    sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    # sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    predictor = SamPredictor(sam)
    
    # Image features encoding
    ref_mask = predictor.set_image(ic_image, ic_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]
    
    # Target feature extraction
    print("======> Obtain Location Prior" )
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)
    
    output_image = []
    
    for test_image in [image1, image2]:
        print("======> Testing Image" )
        test_image = np.array(test_image.convert("RGB"))
        
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
        
        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
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
        mask_colors[final_mask, :] = np.array([[128, 0, 0]])
        output_image.append(Image.fromarray((mask_colors * 0.6 + test_image * 0.4).astype('uint8'), 'RGB'))
    
    return output_image[0].resize((224, 224)), output_image[1].resize((224, 224))


def inference_finetune(ic_image, ic_mask, image1, image2):
    # in context image and mask
    ic_image = np.array(ic_image.convert("RGB"))
    ic_mask = np.array(ic_mask.convert("RGB"))
    
    gt_mask = torch.tensor(ic_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()
    # gt_mask = gt_mask.float().unsqueeze(0).flatten(1)
    
    sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    # sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    for name, param in sam.named_parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)
    
    print("======> Obtain Self Location Prior" )
    # Image features encoding
    ref_mask = predictor.set_image(ic_image, ic_mask)
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
    topk_xy, topk_label, _, _ = point_selection(sim, topk=1)

    print('======> Start Training')
    # Learnable mask weights
    mask_weights = Mask_Weights().cuda()
    # mask_weights = Mask_Weights()
    mask_weights.train()
    train_epoch = 1000
    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=1e-3, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch)

    for train_idx in range(train_epoch):
        # Run the decoder
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        logits_high = logits_high.flatten(1)

        # Weighted sum three-scale masks
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        logits_high = logits_high * weights
        logits_high = logits_high.sum(0).unsqueeze(0)

        dice_loss = calculate_dice_loss(logits_high, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
        loss = dice_loss + focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if train_idx % 10 == 0:
            print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))


    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    weights_np = weights.detach().cpu().numpy()
    print('======> Mask weights:\n', weights_np)

    print('======> Start Testing')
    output_image = []
    
    for test_image in [image1, image2]:
        test_image = np.array(test_image.convert("RGB"))
        
        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()
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
        topk_xy, topk_label, _, _ = point_selection(sim, topk=1)
        
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
        mask_colors[final_mask, :] = np.array([[128, 0, 0]])
        output_image.append(Image.fromarray((mask_colors * 0.6 + test_image * 0.4).astype('uint8'), 'RGB'))
    
    return output_image[0].resize((224, 224)), output_image[1].resize((224, 224))


description = """
<div style="text-align: center; font-weight: bold;">
    <span style="font-size: 18px" id="paper-info">
        [<a href="https://github.com/ZrrSkywalker/Personalize-SAM" target="_blank"><font color='black'>Github</font></a>]
        [<a href="https://arxiv.org/pdf/2305.03048.pdf" target="_blank"><font color='black'>Paper</font></a>]
    </span>
</div>
"""

main = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="pil", label="in context image",),
        gr.Image(type="pil", label="in context mask"),
        gr.Image(type="pil", label="test image1"),
        gr.Image(type="pil", label="test image2"),  
    ],
    outputs=[
        gr.outputs.Image(type="pil", label="output image1"),
        gr.outputs.Image(type="pil", label="output image2"),
    ],
    allow_flagging="never",
    title="Personalize Segment Anything Model with 1 Shot",
    description=description,
    examples=[
        ["./examples/cat_00.jpg", "./examples/cat_00.png", "./examples/cat_01.jpg", "./examples/cat_02.jpg"],
        ["./examples/colorful_sneaker_00.jpg", "./examples/colorful_sneaker_00.png", "./examples/colorful_sneaker_01.jpg", "./examples/colorful_sneaker_02.jpg"],
        ["./examples/duck_toy_00.jpg", "./examples/duck_toy_00.png", "./examples/duck_toy_01.jpg", "./examples/duck_toy_02.jpg"],
    ]
)

main_scribble = gr.Interface(
    fn=inference_scribble,
    inputs=[
        gr.ImageMask(label="[Stroke] Draw on Image", type="pil"),
        gr.Image(type="pil", label="test image1"),
        gr.Image(type="pil", label="test image2"),  
    ],
    outputs=[
        gr.outputs.Image(type="pil", label="output image1"),
        gr.outputs.Image(type="pil", label="output image2"),
    ],
    allow_flagging="never",
    title="Personalize Segment Anything Model with 1 Shot",
    description=description,
    examples=[
        ["./examples/cat_00.jpg", "./examples/cat_01.jpg", "./examples/cat_02.jpg"],
        ["./examples/colorful_sneaker_00.jpg", "./examples/colorful_sneaker_01.jpg", "./examples/colorful_sneaker_02.jpg"],
        ["./examples/duck_toy_00.jpg", "./examples/duck_toy_01.jpg", "./examples/duck_toy_02.jpg"],
    ]
)

main_finetune = gr.Interface(
    fn=inference_finetune,
    inputs=[
        gr.Image(type="pil", label="in context image"),
        gr.Image(type="pil", label="in context mask"),
        gr.Image(type="pil", label="test image1"),
        gr.Image(type="pil", label="test image2"),  
    ],
    outputs=[
        gr.components.Image(type="pil", label="output image1"),
        gr.components.Image(type="pil", label="output image2"),
    ],
    allow_flagging="never",
    title="Personalize Segment Anything Model with 1 Shot",
    description=description,
    examples=[
        ["./examples/cat_00.jpg", "./examples/cat_00.png", "./examples/cat_01.jpg", "./examples/cat_02.jpg"],
        ["./examples/colorful_sneaker_00.jpg", "./examples/colorful_sneaker_00.png", "./examples/colorful_sneaker_01.jpg", "./examples/colorful_sneaker_02.jpg"],
        ["./examples/duck_toy_00.jpg", "./examples/duck_toy_00.png", "./examples/duck_toy_01.jpg", "./examples/duck_toy_02.jpg"],
    ]
)


demo = gr.Blocks()
with demo:
    gr.TabbedInterface(
        [main, main_scribble, main_finetune], 
        ["Personalize-SAM", "Personalize-SAM-Scribble", "Personalize-SAM-F"],
    )

demo.launch(share=True)