import torch

def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition


class GenerateOurs:
    def __init__(self, model, normalize_self_attention, apply_self_in_rule_10):
        self.model = model
        self.normalize_self_attention = normalize_self_attention 
        self.apply_self_in_rule_10 = apply_self_in_rule_10
    

    def handle_self_attention_image(self, cam, grad):
        # cam = blk.attention.self.get_attention_map().detach()
        # grad = blk.attention.self.get_attn_gradients().detach()
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i += R_i_i_add
        self.R_i_t += R_i_t_add


    def handle_self_attention_lang(self, cam, grad):
        # cam = blk.attention.self.get_attention_map().detach()
        # grad = blk.attention.self.get_attn_gradients().detach()
        # print(grad.shape, cam.shape)
        cam = avg_heads(cam, grad)
        # print(self.R_t_t[0])
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t += R_t_t_add
        self.R_t_i += R_t_i_add


    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam
    
    
    def generate_attn_gradcam (self, text_tokens, image_tokens, device):
        blk = self.model.transformer.blocks[-1]
        grad = blk.attn.get_attn_gradients().detach()
        cam = blk.attn.get_attention_map().detach()
        cam = self.gradcam(cam, grad)

        self.R_t_t = cam[:text_tokens, :text_tokens]
        self.R_t_i = cam[text_tokens:, text_tokens:]

        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i


    def generate_relevance_maps (self, text_tokens, image_tokens, device):

        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        self.R = torch.eye(image_tokens + text_tokens, image_tokens + text_tokens).to(device)

        count = 0
        for blk in self.model.transformer.blocks:
            grad = blk.attn.get_attn_gradients().detach()
            cam = blk.attn.get_attention_map().detach()

            cam = avg_heads(cam, grad)

            self.R = self.R + cam @ self.R

            # grad_lang = grad[:, :, : text_tokens, : text_tokens]
            # grad_image = grad[:, :, text_tokens: , text_tokens:]

            # cam_lang = cam[:, :, : text_tokens, : text_tokens]
            # cam_image = cam[:, :, text_tokens: , text_tokens:]

            # self.handle_self_attention_image(cam_image, grad_image)
            # self.handle_self_attention_lang(cam_lang, grad_lang)
            # print(f"COUNT: {count}")
            count += 1


        self.R_t_t = self.R[:text_tokens, :text_tokens]
        self.R_t_i = self.R[text_tokens:, text_tokens:]


        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i


