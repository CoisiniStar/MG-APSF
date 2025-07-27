import torch.nn as nn
import torch
import math
from Model.base_model import BaseModel
from Model.GATLayer_multimodal import HeteroGraphClassifier
import dgl
import numpy as np
from Model.AKAN import AKAN
from Model.Graph_learning import NGR
from Model.Event_GModel import HGC_Model
import argparse
from parse_config import ConfigParser
import sys
from torch.utils.data import DataLoader
from Combinated_Dataset import *
from tqdm import tqdm

def actor_critic_loss(self, rewards_steps, act_probs_steps, state_values_steps, classifier_loss, actor_loss_list,
                      critic_loss_list):
    num_steps = len(rewards_steps)
    for i in range(num_steps):
        shape0 = state_values_steps[i].shape[0]
        shape1 = state_values_steps[i].shape[1]
        shape2 = self.config['topk'][i]  # baseline: (128,15,1)
        baseline = state_values_steps[i] if i >= 1 else state_values_steps[i][:, :, None]
        if i == num_steps - 1:  # i=2 时,
            terminal_reward = self.config['alpha1'] * (
                        1 - classifier_loss.detach())  # rewards_steps(list:3->tensor(128,5),tensor(128,15),tensor(128,30));
            td_error = self.config['alpha2'] * rewards_steps[i] + ((1 - self.config['alpha2']) * terminal_reward)[:,
                                                                  None] - baseline.expand(shape0, shape1,
                                                                                          shape2).reshape(shape0,
                                                                                                          shape1 * shape2)
        else:
            td_error = self.config['alpha2'] * rewards_steps[i] + self.config['gamma'] * state_values_steps[
                i + 1].squeeze(-1) - baseline.expand(shape0, shape1, shape2).reshape(shape0, shape1 * shape2)
        actor_loss = - torch.log(act_probs_steps[i]) * td_error.detach()
        critic_loss = td_error.pow(2)
        actor_loss_list.append(actor_loss.mean())
        critic_loss_list.append(critic_loss.mean())


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.nn.functional.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    接收模型大小和注意力头数
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # 我们假设d_v总是等于d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) 对所有投影进行批处理，从d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) 对批处理中的所有投影向量应用注意力
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 使用view"连接"并应用最终的线性层
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn



class SpaceDetection(nn.Module):
    def __init__(self, device, model_dim, bert_dim=768):   # model_dim -> out_dim
        super(SpaceDetection, self).__init__()
        self.device = device
        self.linear_text = nn.Sequential(
            nn.Linear(bert_dim, bert_dim // 2),
            nn.ReLU(),
            nn.Linear(bert_dim // 2, model_dim)
        ).to(self.device)
        self.ln1 = nn.LayerNorm(normalized_shape=model_dim).to(self.device)
        self.ln2 = nn.LayerNorm(normalized_shape=model_dim).to(self.device)
        # self.bn1 = nn.BatchNorm1d(model_dim).to(self.device)
        # self.bn2 = nn.BatchNorm1d(model_dim).to(self.device)
        self.bn3 = nn.BatchNorm1d(3).to(self.device)

        self.linear_evidence=nn.Sequential(
            nn.Linear(bert_dim, bert_dim//2),
            nn.ReLU(),
            nn.Linear(bert_dim//2,model_dim)
        ).to(self.device)
        self.linear_compare=nn.Sequential(
            nn.Linear(model_dim,model_dim//2), # 768, 128
            nn.ReLU(),
            nn.Linear(model_dim//2, 2) # 128 , 2
        ).to(self.device)
        self.lineartime = nn.Linear(1,1).to(self.device)
        self.linearout = nn.Sequential(
            nn.Linear(bert_dim+1, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 2)
        )
        self.activation = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid()
        )
        self.multi_head_attn = MultiHeadedAttention(h=8, d_model=model_dim, dropout=0.2)
    # text: (32, 109, 768); evidence: (32, 511, 768)  time_info: tensor(32,)
    def forward(self, text, evidence, time_info):
        # 获取text的shape
        text = text.to(self.device)
        evidence = evidence.to(self.device)
        time_info = time_info.float().to(self.device)

        batch_size, seq_length, _ = evidence.shape
        evidence = self.ln1(self.linear_evidence(evidence))

        text = self.ln2(self.linear_text(text))

        time_info = self.lineartime(time_info.unsqueeze(-1))   # double or float

        # combine_feature = torch.cat([evidence, text, text-evidence, text*evidence],  dim=-1)
        output, _ = self.multi_head_attn(
            query=text,  # (32, 397, 128)
            key=evidence,  # (32, 511, 128)
            value=evidence  # (32, 511, 128)
        )
        # (32, 128)
        aggregated = output.mean(dim=1)  # (batch, model_dim)
        out = self.linear_compare(aggregated)  # (batch, 2)
        time_feat = self.lineartime(time_info)  # (batch, 1)
        combined = torch.cat([out, time_feat], dim=-1)  # (batch, 3)
        bn_combined = self.bn3(combined)
        # out_model = self.linearout(bn_combined)

        return bn_combined  # Tensor(batch_size, 3)




class Detection_Module(nn.Module):
    def __init__(self, config, doc_entity_dict, entity_doc_dict, doc_feature_dir, entity_adj, relation_adj, entity_id_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, device, image_embedding, sentence_embedding, sentence_knowledge, image_knowledge, text_graph, in_feats_embedding=[768, 512], out_feats_embedding=[512, 256], classifier_dims=[256], dropout_p=0.6, n_classes=2):
        super(Detection_Module, self).__init__()
        self.device = device
        self.AKAN_device = config["AKAN_device"]
        self.gnn_device = config["gnn_device"]

        self.config = config
        self.doc_entity_dict = doc_entity_dict  # 这里的格式为：字典类型，键是新闻ID
        self.entity_doc_dict = entity_doc_dict
        self.doc_feature_dir = doc_feature_dir
        self.entity_adj = entity_adj.to(self.AKAN_device)
        self.relation_adj = relation_adj.to(self.AKAN_device)
        self.entity_id_dict = entity_id_dict
        self.neibor_embedding = neibor_embedding
        # 原有的neibor_embedding直接进行了编码表示，但是这里的编码是使用nn.Embedding来获取的
        # 这里需要替换成编码的文件夹，而不是嵌入层
        # self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        # self.neibor_embedding = neibor_embedding
        self.neibor_num = neibor_num.to(self.AKAN_device)

        self.image_embedding = image_embedding
        self.sentence_embedding = sentence_embedding
        self.sentence_knowledge = sentence_knowledge
        self.image_knowledge = image_knowledge
        self.text_graph = text_graph
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.in_feats_embedding = in_feats_embedding
        self.out_feats_embedding = [512, 256]
        self.classifier_dims = classifier_dims
        self.dropout_p = dropout_p
        self.n_classes = n_classes
        self.model_dim = 128
        self.bert_dim = 768  # BERT feature dimension
        self.bn = nn.BatchNorm1d(8+8+2+2+3).to(self.device)

        self.AKAN = AKAN(config=self.config, doc_entity_dict=self.doc_entity_dict, entity_doc_dict=self.entity_doc_dict,
                        doc_feature_dir=self.doc_feature_dir, entity_adj=self.entity_adj,
                        relation_adj=self.relation_adj, entity_id_dict=self.entity_id_dict,
                        neibor_embedding=self.neibor_embedding, neibor_num=self.neibor_num, entity_embedding=self.entity_embedding,
                        relation_embedding=self.relation_embedding, device=self.AKAN_device)
        self.NGR = NGR(config=self.config, image_embedding=self.image_embedding, sentence_embedding=self.sentence_embedding,
        sentence_knowledge=self.sentence_knowledge, image_knowledge=self.image_knowledge, text_graph=self.text_graph, entity_embedding=self.entity_embedding, device=self.AKAN_device)
        self.gnn_model = HGC_Model(
        config=self.config,
        in_feats_embedding=[768, 512],
        out_feats_embedding=[512, 256],
        classifier_dims=[128],
        dropout_p=0.6,
        n_classes=2)
        self.TEXT = nn.Sequential(
            nn.Linear(768, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, 8)
        ).to(self.device)
        self.IMAGE = nn.Sequential(
            nn.Linear(768, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, 8)
        ).to(self.device)
        self.Space = SpaceDetection(device =  self.device, model_dim=128, bert_dim=768).to(device)  # 128是输出维度，768是BERT的输入维度
        self.classifier = nn.Sequential(
            nn.Linear(8+8+2+2+3, 128),  # 16
            nn.ReLU(),
            nn.Linear(128,2)
        ).to(self.device)
        self.anchor_activation = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        ).to(config["AKAN_device"])
        self.gnn_activation = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        ).to(config["gnn_device"])
        self.space_activation = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, news, batched_img, batched_text, batched_event, batched_evidence, batched_time_info, batched_labels):
        act_probs_steps1, state_values_steps1, rewards_steps1, anchor_graph1, anchor_relation1 = self.AKAN(news)
        _, _, logits_anchor_graph = self.NGR(adaptive_graph=anchor_graph1, adaptive_relation=anchor_relation1,
                                              news=news)
        score_anchor_graph = self.anchor_activation(logits_anchor_graph)

        logits_gnn_model = self.gnn_model(batched_img, batched_text, batched_event)
        logits_gnn_model = torch.cat(logits_gnn_model, dim=0)
        score_gnn_model = self.gnn_activation(logits_gnn_model)


        logits_space_model = self.Space(batched_text, batched_evidence, batched_time_info)  # Tensor(32,3)
        score_space_model = self.space_activation(logits_space_model)

        batched_text = batched_text.to(self.device)
        text_feature = self.TEXT(batched_text)
        batched_img = batched_img.to(self.device)
        image_feature = self.IMAGE(batched_img)  # 8+8+2+2+3
        print(text_feature.mean(dim=-1).shape)
        out_feature = torch.cat([text_feature.mean(dim=1).to(self.device), image_feature.mean(dim=1).to(self.device), score_anchor_graph.to(self.device) * logits_anchor_graph.to(self.device), score_gnn_model.to(self.device) * logits_gnn_model.to(self.device), score_space_model.to(self.device) * logits_space_model.to(self.device)], dim=-1)
        out_feature = self.bn(out_feature)   # 这里text_feature和image_feature都是三维的
        output = self.classifier(out_feature)
        # 需要在train函数中加入actor_critic_loss函数
        return output, score_anchor_graph, score_gnn_model, score_space_model, rewards_steps1, act_probs_steps1, state_values_steps1

class Detection_Module_tsne(nn.Module):
    def __init__(self, config, doc_entity_dict, entity_doc_dict, doc_feature_dir, entity_adj, relation_adj, entity_id_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding, device, image_embedding, sentence_embedding, sentence_knowledge, image_knowledge, text_graph, in_feats_embedding=[768, 512], out_feats_embedding=[512, 256], classifier_dims=[256], dropout_p=0.6, n_classes=2):
        super(Detection_Module_tsne, self).__init__()
        self.device = device
        self.AKAN_device = config["AKAN_device"]
        self.gnn_device = config["gnn_device"]

        self.config = config
        self.doc_entity_dict = doc_entity_dict  # 这里的格式为：字典类型，键是新闻ID
        self.entity_doc_dict = entity_doc_dict
        self.doc_feature_dir = doc_feature_dir
        self.entity_adj = entity_adj.to(self.AKAN_device)
        self.relation_adj = relation_adj.to(self.AKAN_device)
        self.entity_id_dict = entity_id_dict
        self.neibor_embedding = neibor_embedding
        # 原有的neibor_embedding直接进行了编码表示，但是这里的编码是使用nn.Embedding来获取的
        # 这里需要替换成编码的文件夹，而不是嵌入层
        # self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        # self.neibor_embedding = neibor_embedding
        self.neibor_num = neibor_num.to(self.AKAN_device)

        self.image_embedding = image_embedding
        self.sentence_embedding = sentence_embedding
        self.sentence_knowledge = sentence_knowledge
        self.image_knowledge = image_knowledge
        self.text_graph = text_graph
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.in_feats_embedding = in_feats_embedding
        self.out_feats_embedding = [512, 256]
        self.classifier_dims = classifier_dims
        self.dropout_p = dropout_p
        self.n_classes = n_classes
        self.model_dim = 128
        self.bert_dim = 768  # BERT feature dimension
        self.bn = nn.BatchNorm1d(8+8+2+2+3).to(self.device)

        self.AKAN = AKAN(config=self.config, doc_entity_dict=self.doc_entity_dict, entity_doc_dict=self.entity_doc_dict,
                        doc_feature_dir=self.doc_feature_dir, entity_adj=self.entity_adj,
                        relation_adj=self.relation_adj, entity_id_dict=self.entity_id_dict,
                        neibor_embedding=self.neibor_embedding, neibor_num=self.neibor_num, entity_embedding=self.entity_embedding,
                        relation_embedding=self.relation_embedding, device=self.AKAN_device)
        self.NGR = NGR(config=self.config, image_embedding=self.image_embedding, sentence_embedding=self.sentence_embedding,
        sentence_knowledge=self.sentence_knowledge, image_knowledge=self.image_knowledge, text_graph=self.text_graph, entity_embedding=self.entity_embedding, device=self.AKAN_device)
        self.gnn_model = HGC_Model(
        config=self.config,
        in_feats_embedding=[768, 512],
        out_feats_embedding=[512, 256],
        classifier_dims=[128],
        dropout_p=0.6,
        n_classes=2)
        self.TEXT = nn.Sequential(
            nn.Linear(768, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, 8)
        ).to(self.device)
        self.IMAGE = nn.Sequential(
            nn.Linear(768, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, 8)
        ).to(self.device)
        self.Space = SpaceDetection(device =  self.device, model_dim=128, bert_dim=768).to(device)  # 128是输出维度，768是BERT的输入维度
        self.classifier = nn.Sequential(
            nn.Linear(8+8+2+2+3, 128),  # 16
            nn.ReLU(),
            nn.Linear(128,2)
        ).to(self.device)
        self.anchor_activation = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        ).to(config["AKAN_device"])
        self.gnn_activation = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        ).to(config["gnn_device"])
        self.space_activation = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, news, batched_img, batched_text, batched_event, batched_evidence, batched_time_info, batched_labels):
        act_probs_steps1, state_values_steps1, rewards_steps1, anchor_graph1, anchor_relation1 = self.AKAN(news)
        _, _, logits_anchor_graph = self.NGR(adaptive_graph=anchor_graph1, adaptive_relation=anchor_relation1,
                                              news=news)
        score_anchor_graph = self.anchor_activation(logits_anchor_graph)

        logits_gnn_model = self.gnn_model(batched_img, batched_text, batched_event)
        logits_gnn_model = torch.cat(logits_gnn_model, dim=0)
        score_gnn_model = self.gnn_activation(logits_gnn_model)


        logits_space_model = self.Space(batched_text, batched_evidence, batched_time_info)  # Tensor(32,3)
        score_space_model = self.space_activation(logits_space_model)

        batched_text = batched_text.to(self.device)
        text_feature = self.TEXT(batched_text)
        batched_img = batched_img.to(self.device)
        image_feature = self.IMAGE(batched_img)  # 8+8+2+2+3
        print(text_feature.mean(dim=-1).shape)
        out_feature = torch.cat([text_feature.mean(dim=1).to(self.device), image_feature.mean(dim=1).to(self.device), score_anchor_graph.to(self.device) * logits_anchor_graph.to(self.device), score_gnn_model.to(self.device) * logits_gnn_model.to(self.device), score_space_model.to(self.device) * logits_space_model.to(self.device)], dim=-1)
        out_feature = self.bn(out_feature)   # 这里text_feature和image_feature都是三维的
        tsne_feature = out_feature
        output = self.classifier(out_feature)
        # 需要在train函数中加入actor_critic_loss函数
        return output, score_anchor_graph, score_gnn_model, score_space_model, rewards_steps1, act_probs_steps1, state_values_steps1, tsne_feature










