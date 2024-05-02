# import gradio as gr
import torch
import copy
import time
import requests
import io
import numpy as np
import re
import json
import urllib.request
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import diags
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pymatting.util.util import row_sum

# import ipdb

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.transforms import pixelbert_transform, pixelbert_transform_randaug
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
import sys

import cv2


IMAGE_SIZE = 224

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 1,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 0,
        "arc": 0,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    with urllib.request.urlopen(
        "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
    ) as url:
        id2ans = json.loads(url.read().decode())

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    model = ViLTransformerSS(_config)
    model.setup("test")
    model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    def infer(url, text):
        try:
            if "http" in url:
                res = requests.get(url)
                image = Image.open(io.BytesIO(res.content)).convert("RGB")
            else:
                image = Image.open(url)
            # orig_shape = np.array(image).shape
            # img = pixelbert_transform(size=IMAGE_SIZE)(image)
            img = pixelbert_transform_randaug(size=IMAGE_SIZE)(image)
            # print("pixelberted image shape: {}".format(img.shape))
            img = img.unsqueeze(0).to(device)

        except:
            return False

        batch = {"text": [text], "image": [img]}

        with torch.no_grad():
            encoded = tokenizer(batch["text"])
            text_tokens = tokenizer.tokenize(batch["text"][0])
            print(text_tokens)
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
            ret = model.infer(batch)
            vqa_logits = model.vqa_classifier(ret["cls_feats"])

        answer = id2ans[str(vqa_logits.argmax().item())]
        # print("cls features shape: {}".format(ret['image_feats'][0].shape))
        # print(np.array(image), answer)
        # return [np.array(image), answer], ret['image_feats'][0], orig_shape
        # return [np.array(image), answer], ret['image_feats'][0], img
        return answer, ret['merged_feats'], img, text_tokens


    # question = "What is the colour of her pants?"
    # question = "Where is Vaishnavi?"
    # question = "Did he wear eye glasses?"
    # question = "Is there an owl?"
    # question = "Is the man swimming?"
    # question = "What animals are shown?"
    question = "What animal hat did she wear?"
    # question = "What is the color of the flowers?"
    # question = "How many windows are there?"
    

    # result, image_feats, text_feats, image, text_tokens = infer('easy_test.jpg', 'What is the colour of the ball?')
    # result, merged_feats, image, text_tokens = infer('images/weird_dj.jpg', question)
    # result, merged_feats, image, text_tokens = infer('images/clock_owl.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/swim.jpg', question)
    result, merged_feats, image, text_tokens = infer('images/nee-sama.jpeg', question)
    # result, merged_feats, image, text_tokens = infer('../../nii_depressed.jpg', question)
    # result, merged_feats, image, text_tokens = infer('images/skii.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/cows.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer("https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg", question)
    PATCH_SIZE = 32

    print(f"Shape of merged feats: {merged_feats.shape}")
    text_length = len(text_tokens) + 2 ## [CLS] and [SEP]


    print(f"QUESTION: {question}")
    print("Answer: {}".format(result))
    pp_img_shape = (image.shape[2], image.shape[3])
    # print("Feature shape: {} | orig_img shape: {}".format(feats.shape, pp_img_shape))
    # sys.exit()

    def get_eigen (feats, modality):
        feats = F.normalize(feats.squeeze(dim = 0), p = 2, dim = -1)

        W_feat = (feats @ feats.T)

        W_feat = (W_feat * (W_feat > 0))
        W_feat = W_feat / W_feat.max() 
        W_feat = W_feat.cpu().numpy()

        def get_diagonal (W):
            D = row_sum(W)
            D[D < 1e-12] = 1.0  # Prevent division by zero.
            D = diags(D)
            return D
        
        D = np.array(get_diagonal(W_feat).todense())

        L = D - W_feat
        # print(L)
        # print("here")
        # L[ np.isnan(L) ] = 0
        # L[ L == np.inf ] = 0
        try:
            eigenvalues, eigenvectors = eigs(L, k = 5, which = 'LM', sigma = 0, M = D)
        except:
            eigenvalues, eigenvectors = eigs(L, k = 5, which = 'SM', sigma = 0, M = D)
        


        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
        if modality == "image":
            return eigenvectors[1][text_length: ]
        else:
            return eigenvectors[1][ :text_length]


    # text_relevance = torch.abs(get_eigen(text_feats[1:-1]))
    # text_relevance = get_eigen(text_feats[1:-1])

    # # dim = int(image_relevance.numel() ** 0.5)
    # image_relevance = get_eigen(image_feats[1:, :])

    text_relevance = get_eigen(merged_feats, "text")[1:-1]
    image_relevance = get_eigen(merged_feats, "image")[1:]
    # print()
    
    image_relevance = image_relevance.reshape(1, 1, pp_img_shape[0]//PATCH_SIZE, pp_img_shape[1]//PATCH_SIZE)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=pp_img_shape, mode='bilinear')
    image_relevance = image_relevance.reshape(pp_img_shape[0], pp_img_shape[1]).cpu().numpy()
    # image_relevance = image_relevance.reshape(pp_img_shape[0]//PATCH_SIZE, pp_img_shape[1]//PATCH_SIZE).cpu().numpy()

    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())


    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam


    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
    axs[0].imshow(vis)
    axs[0].axis('off')
    axs[0].set_title('Spectral Approach Image Relevance')

    ti = axs[1].imshow(text_relevance.unsqueeze(dim = 0).numpy())
    axs[1].set_title("Spectral Approach Word Impotance")
    plt.sca(axs[1])
    plt.xticks(np.arange(len(text_tokens)), text_tokens)
    # plt.sca(axs[1])
    plt.colorbar(ti, orientation = "horizontal", ax = axs[1])
    # plt.imshow(vis)
    plt.show()










    # feats = feats.unsqueeze(dim = 0)
    # print("Answer: {}".format(result[1]))
    # print("Feature shape: {} | orig_shape: {}".format(feats.shape, orig_shape))
    # feats = F.normalize(feats, p = 2, dim = -1)
    # # H_patch, W_patch = orig_shape[0]//PATCH_SIZE, orig_shape[1]//PATCH_SIZE
    # # print("H_patch, W_patch: {}, {}".format(H_patch, W_patch))
    # # H_pad_lr, W_pad_lr = H_patch, W_patch

    # H_patch, W_patch = orig_shape[0] // PATCH_SIZE, orig_shape[1] // PATCH_SIZE
    # H_pad, W_pad = H_patch * PATCH_SIZE, W_patch * PATCH_SIZE
    # H_pad_lr, W_pad_lr = H_pad // PATCH_SIZE, W_pad // PATCH_SIZE

    # if (H_patch, W_patch) != (H_pad_lr, W_pad_lr):
    #     feats = F.interpolate(
    #         feats.T.reshape(1, -1, H_patch, W_patch), 
    #         size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
    #     ).reshape(-1, H_pad_lr * W_pad_lr).T

    # # feats = F.interpolate(
    # #         feats.T.reshape(1, -1,  H_patch, W_patch), 
    # #         size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
    # # ).reshape(-1, H_pad_lr * W_pad_lr).T

    # W_feat = (feats @ feats.T)

    # W_feat = (W_feat * (W_feat > 0))
    # W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
    # W_feat = W_feat.cpu().numpy()


    # def get_diagonal (W):
    #     D = row_sum(W)
    #     D[D < 1e-12] = 1.0  # Prevent division by zero.
    #     D = diags(D)
    #     return D
    
    # D = np.array(get_diagonal(W_feat).todense())

    # L = D - W_feat
    # eigenvalues, eigenvectors = eigsh(L, k = 5, which = 'LM', sigma = 0)
    # eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    # # print(eigenvectors[1].shape)
    # fiedel_ev = eigenvectors[1].numpy().reshape(H_patch, W_patch)
    # plt.imshow(fiedel_ev)
    # plt.show()
    
