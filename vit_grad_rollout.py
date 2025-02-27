import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def grad_rollout(attentions, gradients, discard_ratio):
    attentions = attentions[1::2]
    gradients =gradients[::2]
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1,keepdim=True)
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[:, 0 ,1:]
    # mask = result.mean(1)[0]
    # mask = mask.reshape(8,-1)
    # In case of 224x224 image, this brings us from 196 to 14
    # width = int(mask.size(-1)**0.5)
    
    # mask = mask.reshape(width, width).numpy()
    # mask = mask / np.max(mask)
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(16,width, width).numpy()
    # mask = mask / np.max(mask,axis=(1,2))
    mask = mask / np.max(mask, axis=(1, 2), keepdims=True)
    return mask    

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        a,output = self.model(input_tensor)
        # print(a.argmax())
        category_mask = torch.zeros(output.size(),device='cuda')
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()
        print(output.argmax())
        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)