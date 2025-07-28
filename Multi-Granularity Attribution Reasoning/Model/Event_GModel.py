## Importing libraries
import ctypes

# lib_path = "/home/lhz/cuda12.1/lib64/libcusparse.so.12"
# ctypes.CDLL(lib_path)

import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from Model.GATLayer_multimodal import HeteroGraphClassifier



def Graph_DGL_Multi(device, img_embedding, text_embedding, event_embedding):
    image_vec = img_embedding
    image_adjMat = np.ones((img_embedding.shape[0], img_embedding.shape[0])).astype(float)
    image_pos_arr = np.where(image_adjMat > 0.5)
    graph_data = {
        ('image_node', 'image_edge', 'image_node'):
            (torch.tensor(image_pos_arr[0], dtype=torch.int64), torch.tensor(image_pos_arr[1], dtype=torch.int64))
    }
    text_vec = text_embedding
    text_adjMat = np.ones((text_embedding.shape[0], text_embedding.shape[0])).astype(float)
    text_pos_arr = np.where(text_adjMat > 0.5)
    graph_data[('text_node', 'text_edge', 'text_node')] = (torch.tensor(text_pos_arr[0], dtype=torch.int64),
                                                           torch.tensor(text_pos_arr[1], dtype=torch.int64))

    event_vec = event_embedding
    event_adjMat = np.ones((event_embedding.shape[0], event_embedding.shape[0])).astype(float)
    event_pos_arr = np.where(event_adjMat > 0.5)
    graph_data[('event_node', 'event_edge', 'event_node')] = (torch.tensor(event_pos_arr[0], dtype=torch.int64),
                                                                torch.tensor(event_pos_arr[1], dtype=torch.int64))

    g = dgl.heterograph(graph_data)
    g = g.to(device)
    g.nodes["text_node"].data['features'] = text_embedding.clone().detach().requires_grad_(True)
    g.nodes["image_node"].data['features'] = img_embedding.clone().detach().requires_grad_(True)
    g.nodes["event_node"].data['features'] = event_embedding.clone().detach().requires_grad_(True)
    return g


## Hetero-Graph Classifier ##
class HGC_Model(nn.Module):
    def __init__(self, config,
                 in_feats_embedding=[768, 512],
                 out_feats_embedding=[512, 256],
                 classifier_dims=[128],
                 dropout_p=0.6,
                 n_classes=2):
        super(HGC_Model, self).__init__()
        self.config = config
        self.device = self.config["gnn_device"]
        self.Heterographclassifier = HeteroGraphClassifier(
                 config = self.config,
                 in_feats_embedding= in_feats_embedding,
                 out_feats_embedding= out_feats_embedding,
                 classifier_dims=classifier_dims,
                 dropout_p=dropout_p,
                 n_classes=n_classes)

        ## Fake News Detector
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        ).to(self.device)

    def forward(self, img_embeds, text_embeds, event_embeds):
        device = self.device
        img_embeds = img_embeds.to(device)
        text_embeds = text_embeds.to(device)
        event_embeds = event_embeds.to(device)
        split_img = torch.split(img_embeds, 1, dim=0) # (1, x, 768)
        split_txt = torch.split(text_embeds, 1, dim=0)  # (1, y, 768)
        split_event = torch.split(event_embeds, 1, dim=0) # (1, z, 768)
        pred_list = []
        for item in zip(split_img, split_txt, split_event):
            g_m = Graph_DGL_Multi(device, item[0], item[1], item[2])
            deci_m = self.Heterographclassifier.forward(g_m)
            deci_m = deci_m.flatten()
            deci_m = deci_m.reshape(-1,2)
            max_values = torch.max(deci_m, dim=0)[0]
            deci_m = torch.unsqueeze(max_values, dim=0)
            pred_list.append(deci_m)
        return pred_list
