import torch
import gradio as gr
import os 
import numpy as np
import cv2
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

def segment(image):
    DEVICE = 'cuda'
    best_model = torch.load('./best_model.bin')
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    return pr_mask


iface = gr.Interface(fn=segment, inputs="image", outputs="image").launch()
iface.launch()