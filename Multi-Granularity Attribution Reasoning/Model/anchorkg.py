from accelerate.utils import pad_input_tensors

from Model.base_model import BaseModel
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
# from transformers import BertWordPieceTokenizer, BertModel
import os
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class Net(BaseModel):  # Actor-Critic架构的神经网络
    def __init__(self, config, entityid_dict, doc_feature_dir):
        super(Net, self).__init__()
        self.config = config
        self.doc_feature_dir = doc_feature_dir
        self.entityid_dict = entityid_dict

        self.actor_l1 = nn.Linear(self.config['embedding_size'] * 4, self.config['embedding_size'])
        self.actor_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.actor_l3 = nn.Linear(self.config['embedding_size'], 1)

        # self.critic_l1 = nn.Linear(self.config['embedding_size']*3, self.config['embedding_size'])
        self.critic_l2 = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.critic_l3 = nn.Linear(self.config['embedding_size'], 1)

        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-2)

    # state_input: (7, 384);  action_input: (7, 20, 128)
    def forward(self, state_input, action_input):
        if len(state_input.shape) < len(action_input.shape):
            if len(action_input.shape) == 3:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], state_input.shape[2])
            else:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], action_input.shape[2],
                                                 state_input.shape[3])

        # Actor
        actor_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = self.elu(self.actor_l2(actor_x))
        act_probs = self.softmax(self.actor_l3(actor_out))  # out: (batch, 20, 1),(batch, 5, 20, 1),(batch, 15, 20, 1)

        # Critic
        critic_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        values = self.critic_l3(critic_out).mean(dim=-2)  # out: (batch,1), (batch,5,1), (batch,15,1)

        return act_probs, values


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, device):
        super(ProjectionHead, self).__init__()
        layer_dims = [input_dim, 512]
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(layer_dims[i], layer_dims[i + 1]),  # 线性层
                    nn.BatchNorm1d(layer_dims[i + 1]),  # 批量归一化
                    nn.ReLU(),  # ReLU 激活函数
                    nn.Dropout(0.4)
                )
            )
        self.layers = nn.ModuleList(layers).to(device)

    def forward(self, x):
        x = x.to(torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x


class AKAN(BaseModel):
    def __init__(self, config, doc_entity_dict, entity_doc_dict, doc_feature_dir, entity_adj, relation_adj,
                 entity_id_dict, neibor_embedding, neibor_num, entity_embedding, relation_embedding,
                 device):
        super(AKAN, self).__init__()
        self.device = device
        self.config = config
        self.doc_entity_dict = doc_entity_dict  # 这里的格式为：字典类型，键是新闻ID
        self.doc_feature_dir = doc_feature_dir
        self.entity_adj = entity_adj.to(device)
        self.relation_adj = relation_adj.to(device)
        self.entity_id_dict = entity_id_dict

        self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        # 原有的neibor_embedding直接进行了编码表示，但是这里的编码是使用nn.Embedding来获取的
        # 这里需要替换成编码的文件夹，而不是嵌入层
        # self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        # self.neibor_embedding = neibor_embedding
        self.neibor_num = neibor_num.to(device)

        self.MAX_DEPTH = self.config['max_depth']
        self.cos = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-2)
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding).to(device)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding).to(device)
        self.news_compress = nn.Sequential(
            nn.Linear(self.config['doc_embedding_size'], self.config['embedding_size']),
            nn.ELU(),
            nn.Linear(self.config['embedding_size'], self.config['embedding_size']),
            nn.Tanh()
        ).to(device)
        self.entity_compress = nn.Sequential(
            nn.Linear(self.config['entity_embedding_size'], self.config['embedding_size'], bias=False),
            nn.Tanh(),
        ).to(device)
        self.relation_compress = nn.Sequential(
            nn.Linear(self.config['entity_embedding_size'], self.config['embedding_size'], bias=False),
            nn.Tanh(),
        ).to(device)
        self.innews_relation = nn.Embedding(1, self.config['embedding_size']).to(device)
        self.adaptive_embedding_layer = nn.Sequential(
            nn.Linear(self.config['embedding_size'] * 2, self.config['embedding_size'], bias=False),
            nn.Tanh(),
        ).to(device)
        self.adaptive_layer = nn.Sequential(
            nn.Linear(self.config['embedding_size'], self.config['embedding_size'], bias=False),
            nn.ELU(),
            nn.Linear(self.config['embedding_size'], 1, bias=False),
        ).to(device)

        self.policy_net = Net(self.config, self.entity_id_dict, self.doc_feature_dir).to(device)

    def get_news_embedding_batch(self, newsids):  # (batch, 768)  获取一个batch中新闻的编码
        # 确保newsids是可迭代的
        if isinstance(newsids, int):
            newsids = [newsids]

        news_embeddings = []
        for newsid in newsids:
            news_path = os.path.join(self.doc_feature_dir, f"{newsid}.npy")
            news_full_text_path = os.path.join(self.doc_feature_dir, f"{newsid}_full_text.npy")
            if os.path.exists(news_path) and os.path.exists(news_full_text_path):
                news_feature = np.load(news_path)
                news_full_text = np.load(news_full_text_path)
                news_feature = torch.from_numpy(news_feature)
                news_full_text = torch.from_numpy(news_full_text)
                news_full_text = news_full_text
                # print('news_feature.shape:', news_feature.shape, 'news_full_text.shape:', news_full_text.shape)
                # combined = torch.cat((news_feature, news_full_text), dim=0)
                news_embeddings.append(news_full_text)
            else:
                print(f"Feature file not found: {newsid}, filled with zeros.")
                news_embeddings.append(torch.zeros(768))
        # news_embeddings[list[tensor[768]]) 转成 [batch_size, 768]
        news_embeddings = torch.stack(news_embeddings)
        return news_embeddings

    def get_news_entities_batch(self, newsids):  # 包含在当前新闻中的实体
        news_entities = torch.zeros(len(newsids), self.config['news_entity_num'], dtype=torch.long)
        news_relations = torch.zeros(len(newsids), self.config['news_entity_num'], dtype=torch.long)  # 关系都是zero
        for i in range(len(newsids)):
            # print(newsids[i])
            news_entities[i] = self.doc_entity_dict[str(newsids[i])]

        return news_entities.to(self.device), news_relations.to(self.device)

    def get_state_input(self, news_embedding, depth, adaptive_graph, history_entity, history_relation):
        if depth == 0:  # 如果深度为0，则状态值由新闻嵌入和零向量拼接而成，表示初始状态
            state_embedding = torch.cat(
                [news_embedding, torch.zeros([news_embedding.shape[0], 128 * 2], dtype=torch.float32).to(self.device)],
                dim=-1)
        else:
            # 获取历史实体和关系嵌入表示
            history_entity_embedding = self.entity_compress(
                self.entity_embedding(history_entity.to(self.entity_embedding.weight.device)).to(self.device))
            history_relation_embedding = self.relation_compress(
                self.relation_embedding(history_relation.to(self.relation_embedding.weight.device)).to(
                    self.device)) if depth > 1 else self.innews_relation(history_relation)

            state_embedding_new = history_relation_embedding + history_entity_embedding
            state_embedding_new = torch.mean(state_embedding_new, dim=1, keepdim=False)
            adaptive_embedding = self.get_adaptive_graph_embedding(adaptive_graph)  # 获取自适应图嵌入
            state_embedding = torch.cat([news_embedding, adaptive_embedding, state_embedding_new], dim=-1)
        return state_embedding

    def get_adaptive_graph_embedding(self, adaptive_graph):
        adaptive_graph_nodes = torch.cat(adaptive_graph, dim=-1).to(self.entity_adj.device)
        adaptive_graph_nodes_embedding = self.entity_compress(
            self.entity_embedding(adaptive_graph_nodes).to(self.device))
        neibor_entities, neibor_relations = self.get_neighbors(
            adaptive_graph_nodes)  # first-order neighbors for each entity
        neibor_entities_embedding = self.entity_compress(self.entity_embedding(neibor_entities).to(self.device))
        neibor_relations_embedding = self.relation_compress(self.relation_embedding(neibor_relations).to(self.device))
        adaptive_embedding = torch.cat(
            [adaptive_graph_nodes_embedding, torch.sum(neibor_entities_embedding + neibor_relations_embedding, dim=-2)],
            dim=-1)

        adaptive_embedding = self.adaptive_embedding_layer(adaptive_embedding)  # (batch, 50, 128)
        adaptive_embedding_weight = self.softmax(self.adaptive_layer(adaptive_embedding))  # (batch, 50, 1)
        adaptive_embedding = torch.sum(adaptive_embedding * adaptive_embedding_weight, dim=-2)
        return adaptive_embedding

    def get_neighbors(self, entities):
        neighbor_entities = self.entity_adj[entities]
        neighbor_relations = self.relation_adj[entities]
        return neighbor_entities, neighbor_relations

    def get_adaptive_nodes(self, weights, action_id_input, relation_id_input, topk):
        if len(weights.shape) <= 3:
            weights = torch.unsqueeze(weights, 1)
            action_id_input = torch.unsqueeze(action_id_input, 1)
            relation_id_input = torch.unsqueeze(relation_id_input, 1)

        weights = weights.squeeze(-1)
        m = Categorical(weights)
        acts_idx = m.sample(sample_shape=torch.Size([topk]))  # may sample the same position multiple times
        acts_idx = acts_idx.permute(1, 2, 0)
        shape0 = acts_idx.shape[0]
        shape1 = acts_idx.shape[1]
        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2])  # (batch,topk)

        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2])
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1],
                                                  action_id_input.shape[2])
        relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1],
                                                      relation_id_input.shape[2])

        weights = weights.gather(1, acts_idx)
        state_id_input_value = action_id_input.gather(1, acts_idx)  # selected entity id,(batch,topk)
        relation_id_selected = relation_id_input.gather(1, acts_idx)  # selected relation id,(batch,topk)

        weights = weights.reshape(shape0, shape1 * weights.shape[
            1])  # probility for selected (r,e) ,(batch,5), (batch, 15) , (batch, 30)
        state_id_input_value = state_id_input_value.reshape(shape0, shape1 * state_id_input_value.shape[1])
        relation_id_selected = relation_id_selected.reshape(shape0, shape1 * relation_id_selected.shape[1])
        return weights, state_id_input_value, relation_id_selected

    """---新闻相关奖励---"""

    def get_reward(self, news_embedding, adaptive_nodes):
        # 新加入的实体嵌入是通过对E_t中的实体的平均得到的
        neibor_news_embedding_avg = self.get_neiborhood_news_embedding_batch(news_embedding, adaptive_nodes)
        """构建一个反向索引，用于追踪所有在标题中提及某个实体e的新闻"""
        sim_reward = self.get_sim_reward_batch(news_embedding, neibor_news_embedding_avg)
        reward = sim_reward
        return reward

    def pad_and_stack(self, tensor_list, pad_value=0.0, dim=1):
        assert isinstance(tensor_list, list), "The input must be a tensor list"
        assert all(isinstance(t, torch.Tensor) for t in tensor_list), "The element of the list must be tensor type"
        assert all(t.dim() == 3 for t in tensor_list), "dimension of the tensor must be like [1, seq_len, hidden_size]"

        # 获取每个 tensor 的 seq_len
        seq_lens = [t.size(dim) for t in tensor_list]
        max_len = max(seq_lens)

        # 获取 hidden_size
        hidden_size = tensor_list[0].size(2)

        padded_tensors = []
        for t in tensor_list:
            seq_len = t.size(1)
            pad_len = max_len - seq_len

            # pad: (dim_right, dim_left, dim_down, dim_up, ...)，这里 pad 第二维度
            padded_t = F.pad(t, (0, 0, 0, pad_len), value=pad_value)
            padded_tensors.append(padded_t)

        # 拼接成 [batch_size, max_len, hidden_size]
        padded_tensor = torch.cat(padded_tensors, dim=0)

        return padded_tensor

    def get_neiborhood_news_embedding_batch(self, news_embedding,
                                            adaptive_nodes):  # (batch,5,768), doc neiborhood embedding avg for each entity
        # 将这里的neibor_embedding替换为neibor_list中对应的编码
        neibor_news_embedding_avg = self.neibor_embedding(adaptive_nodes.to(self.neibor_embedding.weight.device)).to(
            self.device)
        # self.neibor_list
        neibor_num = self.neibor_num[adaptive_nodes]
        neibor_news_embedding_avg = torch.div((neibor_news_embedding_avg - news_embedding[:, None, :]),
                                              neibor_num[:, :, None])
        return neibor_news_embedding_avg  # (batch, 5/15/30, 768)

    def get_sim_reward_batch(self, news_embedding_batch, neibor_news_embedding_avg_batch):
        cos_rewards = self.cos(news_embedding_batch[:, None, :], neibor_news_embedding_avg_batch)
        return cos_rewards  # (batch, 5/15/30)

    def forward(self, news):  # news: batch and newsid
        depth = 0
        history_entity = []
        history_relation = []

        adaptive_graph = []
        adaptive_relation = []
        act_probs_steps = []  # 策略网络的动作可能性
        state_values_steps = []  # 目标网络的状态值
        rewards_steps = []  # immediate reward
        # 获取一个batch中新闻的编码  list[tensor[x, 768]]
        news_embedding_origin = self.get_news_embedding_batch(news)  # (batch, 768)
        # 在这里判断news_embedding_origin是否是一个list，如果是就转成Tensor，同时保持第一个维度不变

        # if isinstance(news_embedding_origin_3d, list):
        #     news_embedding_origin_3d = self.pad_and_stack(tensor_list=news_embedding_origin_3d)
        # argument 'input' (position 1) must be Tensor, not list
        # news_embedding_origin_3d = news_embedding_origin_3d.to(self.device)  # (batchsize, seq_len, 768)
        news_embedding_origin = news_embedding_origin.to(self.device)  # (batchsize, 768)
        news_embedding = self.news_compress(news_embedding_origin)  # (batch, 128)

        action_id, relation_id = self.get_news_entities_batch(news)  # Current News 中的实体和关系，关系 ID==0，CPU
        action_embedding = self.entity_compress(self.entity_embedding(action_id).to(self.device))  # (batch, 20, 128)
        relation_embedding = self.innews_relation(
            relation_id.to(self.device))  # self.relation_compress(self.relation_embedding(relation_id).to(self.device))

        action_embedding = action_embedding + relation_embedding  # (batch, 20, 128)

        state_input = self.get_state_input(news_embedding, depth, adaptive_graph, history_entity,
                                           history_relation)  # (batch, 384)

        while (depth < self.MAX_DEPTH):
            act_probs, state_values = self.policy_net(state_input, action_embedding)
            topk = self.config['topk'][depth]  # 5, 3, 2
            adaptive_act_probs, adaptive_nodes, adaptive_relations = self.get_adaptive_nodes(act_probs,
                                                                                             action_id.to(self.device),
                                                                                             relation_id.to(
                                                                                                 self.device),
                                                                                             topk)  # take action

            history_entity = adaptive_nodes  # newly adds entities
            history_relation = adaptive_relations
            depth = depth + 1
            act_probs_steps.append(adaptive_act_probs)
            state_values_steps.append(state_values)
            adaptive_graph.append(adaptive_nodes)  # gpu
            adaptive_relation.append(adaptive_relations)

            state_input = self.get_state_input(news_embedding, depth, adaptive_graph, history_entity,
                                               history_relation)  # next state

            action_id, relation_id = self.get_neighbors(adaptive_nodes)
            action_embedding = self.entity_compress(
                self.entity_embedding(action_id).to(self.device)) + self.relation_compress(
                self.relation_embedding(relation_id).to(self.device))

            step_reward = self.get_reward(news_embedding_origin, adaptive_nodes)
            rewards_steps.append(step_reward)

        return act_probs_steps, state_values_steps, rewards_steps, adaptive_graph, adaptive_relation  # 只有act_probs_steps, state_values_steps有梯度

    def warm_train(self, batch):
        loss_fn = nn.BCELoss()
        news_embedding = self.get_news_embedding_batch(batch['newsid'])  # 对新闻内容进行编码
        news_embedding = self.news_compress(news_embedding)  # 新闻内容特征压缩
        path_node_embeddings = self.entity_compress(
            self.entity_embedding(batch['paths']).to(self.device))  # (batch, depth, embedding_size)
        path_edge_embeddings = self.relation_compress(
            self.relation_embedding(batch['edges'][:, 1:]).to(self.device))  # (batch, depth-1, embedding_size)
        innews_relation_embedding = self.innews_relation(
            torch.zeros([batch['paths'].shape[0], 1], dtype=torch.long).to(self.device))  # (batch, 1, embedding_size)
        path_edge_embeddings = torch.cat([innews_relation_embedding, path_edge_embeddings], dim=1)
        path_embeddings = path_node_embeddings + path_edge_embeddings  # (batch, depth, 128)

        batch_act_probs = []
        adaptive_graph = []
        history_entity = []
        history_relation = []
        for i in range(self.MAX_DEPTH):
            state_input = self.get_state_input(news_embedding, i, adaptive_graph, history_entity,
                                               history_relation)  # (batch,256)
            act_probs, _ = self.policy_net(state_input, path_embeddings[:, i, :])  # (batch, 1)
            batch_act_probs.append(act_probs)
            adaptive_graph.append(batch['paths'][:, i:i + 1].to(self.device))
            history_entity = batch['paths'][:, i:i + 1].to(
                self.device)  # same as forward func，state_input only depends on newly added nodes
            history_relation = batch['edges'][:, i:i + 1].to(self.device)

        batch_act_probs = torch.cat(batch_act_probs, dim=1)  # (batch, depth)
        indices = batch['label'] >= -0.5
        labels = batch['label'][indices].to(self.device)
        predicts = batch_act_probs[indices.to(self.device)]
        loss = loss_fn(predicts, labels)

        return loss, predicts, labels



