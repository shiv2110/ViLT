import gradio as gr
import torch
import copy
import time
import requests
import io
import numpy as np
import re
import json
import urllib.request
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pymatting.util.util import row_sum

# import ipdb

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer


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
            # res = requests.get(url)
            # image = Image.open(io.BytesIO(res.content)).convert("RGB")

            image = Image.open(url)
            print("Original image shape: {}".format(np.array(image).shape))
            img = pixelbert_transform(size=384)(image)
            print("pixelberted image shape: {}".format(img.shape))
            img = img.unsqueeze(0).to(device)

        except:
            return False

        batch = {"text": [text], "image": [img]}

        with torch.no_grad():
            encoded = tokenizer(batch["text"])
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
            ret = model.infer(batch)
            vqa_logits = model.vqa_classifier(ret["cls_feats"])

        answer = id2ans[str(vqa_logits.argmax().item())]
        # print("cls features shape: {}".format(ret['image_feats'][0].shape))
        # print(np.array(image), answer)
        return [np.array(image), answer], ret['image_feats'][0]
    


    # inputs = [
    #     gr.inputs.Textbox(
    #         label="Url of an image.",
    #         lines=5,
    #     ),
    #     gr.inputs.Textbox(label="Question", lines=5),
    # ]
    # outputs = [
    #     gr.outputs.Image(label="Image"),
    #     gr.outputs.Textbox(label="Answer"),
    # ]

    # interface = gr.Interface(
    #     fn=infer,
    #     inputs=inputs,
    #     outputs=outputs,
    #     server_name="0.0.0.0",
    #     server_port=8888,
    #     examples=[
    #         [
    #             "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
    #             "What is the color of the flower?",
    #         ],
    #         [
    #             "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_1.png",
    #             "What is the mustache made of?",
    #         ],
    #         [
    #             "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_2.png",
    #             "How many slices of pizza are there?",
    #         ],
    #         [
    #             "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_3.png",
    #             "Does it appear to be rainy?",
    #         ],
    #     ],
    # )

    # interface.launch(debug=True)
    result, feats = infer('nii_depressed.jpg', 'What is the colour of the shirt?')
    # result, feats = infer("https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg", "What is the color of the flower?")
    
    print("Answer: {}".format(result[1]))
    H_patch, W_patch = 16, 12
    H_pad_lr, W_pad_lr = H_patch, W_patch
    feats = F.interpolate(
            feats.T.reshape(1, -1, feats.shape[0], feats.shape[1]), 
            size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
    ).reshape(-1, H_pad_lr * W_pad_lr).T

        ### Feature affinities 
    W_feat = (feats @ feats.T)

    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
    W_feat = W_feat.cpu().numpy()
    
    # temp = np.matmul(image_feat.numpy(), np.transpose(image_feat.numpy()))

    # W = np.where(temp > 0, temp, 0)

    # D = np.zeros(W_feat.shape)
    # for i in range(W_feat.shape[0]):
    #     D[i, i] = np.sum(W_feat[i])

    def get_diagonal (W):
        D = row_sum(W)
        D[D < 1e-12] = 1.0  # Prevent division by zero.
        D = diags(D)
        return D
    
    D = np.array(get_diagonal(W_feat).todense())

    L = D - W_feat
    eigenvalues, eigenvectors = eigsh(L, k = 5, sigma = 0, which = 'LM')
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    # print(eigenvectors[1].shape)
    fiedel_ev = eigenvectors[1].numpy().reshape(16, 12)
    plt.imshow(fiedel_ev)
    plt.show()
    
