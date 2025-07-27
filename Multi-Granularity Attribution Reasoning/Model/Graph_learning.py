import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.autograd import Variable
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
from torch.distributions import Bernoulli, Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from Model.base_model import BaseModel
import numpy as np

VERY_SMALL_NUMBER = 1e-12
INF = 1e20


class GAT(BaseModel):
    def __init__(self, in_features, out_features, heads, dropout, alpha, concat=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def forward(self, hidden_state, adjacent_matrix=None):
        """
        :param hidden_state: [batch_size, seq_len, in_features]
        :param adjacent_matrix: [batch_size, seq_len, seq_len]
        """
        _x = self.linear(hidden_state)
        e = self._para_attentional_mechanism_input(_x)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adjacent_matrix > 0.5, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, _x)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _para_attentional_mechanism_input(self, Wh):
        """
        :param Wh: [batch_size, seq_len, out_features]
        """
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphLearner(BaseModel):
    def __init__(self, input_size, hidden_size, graph_type, top_k=None, epsilon=None, num_pers=1,
                 metric_type="attention", feature_denoise=True, device=None,
                 temperature=0.1):
        super(GraphLearner, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_pers = num_pers
        self.graph_type = graph_type
        self.top_k = top_k
        self.epsilon = epsilon
        self.metric_type = metric_type
        self.feature_denoise = feature_denoise
        self.temperature = temperature

        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList(
                [nn.Linear(self.input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, -num_pers))
        elif metric_type == 'weighted_cosine':
            # self.weight_tensor = torch.Tensor(num_pers, self.input_size)
            self.weight_tensor = torch.Tensor(num_pers, self.input_size).to(device)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))
        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.leakyrelu = nn.LeakyReLU(0.2)
            print('[ GAT_Attention GraphLearner]')
        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(input_size, hidden_size, bias=False)
        elif metric_type == 'cosine':
            pass
        elif metric_type == 'mlp':
            self.lin = nn.Linear(self.input_size * 2, 1)
        elif metric_type == 'multi_mlp':
            self.linear_sims1 = nn.ModuleList(
                [nn.Linear(self.input_size, hidden_size, bias=False) for _ in range(num_pers)]).to(device)
            self.linear_sims2 = nn.ModuleList(
                [nn.Linear(self.hidden_size, hidden_size, bias=False) for _ in range(num_pers)]).to(device)
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))
        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(input_size, init_strategy="constant")

        print('[ Graph Learner metric type: {}, Graph Type: {} ]'.format(metric_type, self.graph_type))

    def reset_parameters(self):
        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(self.input_size, init_strategy="constant")
        if self.metric_type == 'attention':
            for module in self.linear_sims:
                module.reset_parameters()
        elif self.metric_type == 'weighted_cosine':
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        elif self.metric_type == 'gat_attention':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()
        elif self.metric_type == 'kernel':
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.init.xavier_uniform_(self.weight)
        elif self.metric_type == 'transformer':
            self.linear_sim1.reset_parameters()
            self.linear_sim2.reset_parameters()
        elif self.metric_type == 'cosine':
            pass
        elif self.metric_type == 'mlp':
            self.lin1.reset_parameters()
            self.lin2.reset_parameters()
        elif self.metric_type == 'multi_mlp':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()
        else:
            raise ValueError('Unknown metric_type: {}'.format(self.metric_type))

    def forward(self, node_features, node_mask=None):
        if self.feature_denoise:
            masked_features = self.mask_feature(node_features)
            learned_adj = self.learn_adj(masked_features, ctx_mask=node_mask)
            return masked_features, learned_adj
        else:
            learned_adj = self.learn_adj(node_features, ctx_mask=node_mask)
            return node_features, learned_adj

    """
        learn_adj：实现 Top-K 剪枝
        支持多种相似度度量（metric_type参数）：
            - weighted_cosine：对应公式5的加权余弦
            - gat_attention：GAT 注意力机制
            - kernel：核方法
        使用 build_knn_neighbourhood() 实现公式(10)的 Top-K 操作
    """

    def learn_adj(self, context, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)
        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        context = context.to(self.device)
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0

        elif self.metric_type == 'transformer':
            Q = self.linear_sim1(context)
            attention = torch.matmul(Q, Q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            markoff_value = -INF

        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF
            # markoff_value = 0

        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis ** 2))

            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0
        elif self.metric_type == 'mlp':
            seq_len = context.size(1)
            context_fc = context.unsqueeze(1).repeat(1, seq_len, 1, 1)
            context_bc = context.unsqueeze(2).repeat(1, 1, seq_len, 1)
            attention = F.sigmoid(self.lin(torch.cat([context_fc, context_bc], dim=-1)).squeeze())
            markoff_value = 0
        elif self.metric_type == 'multi_mlp':
            attention = 0
            for _ in range(self.num_pers):
                context_fc = torch.relu(self.linear_sims2[_](torch.relu(self.linear_sims1[_](context))))
                attention += F.sigmoid(torch.matmul(context_fc, context_fc.transpose(-1, -2)))

            attention /= self.num_pers
            markoff_value = -INF

        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.graph_type == 'epsilonNN':
            assert self.epsilon is not None
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        elif self.graph_type == 'KNN':
            assert self.top_k is not None
            attention = self.build_knn_neighbourhood(attention, self.top_k, markoff_value)
        elif self.graph_type == 'prob':
            attention = self.build_prob_neighbourhood(attention, self.epsilon, temperature=self.temperature)
        else:
            raise ValueError('Unknown graph_type: {}'.format(self.graph_type))
        if self.graph_type in ['KNN', 'epsilonNN']:
            if self.metric_type in ('kernel', 'weighted_cosine'):
                assert attention.min().item() >= 0
                attention = attention / torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            elif self.metric_type == 'cosine':
                attention = (attention > 0).float()
                attention = self.normalize_adj(attention)
            elif self.metric_type in ('transformer', 'attention', 'gat_attention'):
                attention = torch.softmax(attention, dim=-1)

        return attention

    def normalize_adj(mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    # Top-K 操作
    def build_knn_neighbourhood(self, attention, top_k, markoff_value):
        top_k = min(top_k, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, top_k, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
        weighted_adjacency_matrix = weighted_adjacency_matrix.to(self.device)

        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        attention = torch.sigmoid(attention)
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def build_prob_neighbourhood(self, attention, epsilon=0.1, temperature=0.1):
        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(attention.device),
                                                     probs=attention).rsample()
        mask = (weighted_adjacency_matrix > epsilon).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def mask_feature(self, x, use_sigmoid=True, marginalize=True):
        feat_mask = (torch.sigmoid(self.feat_mask) if use_sigmoid else self.feat_mask).to(self.device)
        if marginalize:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
            z = torch.normal(mean=mean_tensor, std=std_tensor).to(self.device)
            x = x + z * (1 - feat_mask)
        else:
            x = x * feat_mask
        return x


"""
    对应于论文中的3.3节 模态注意力分层池化

"""


class DiffPool(nn.Module):

    def __init__(self, config, feature_size, output_dim, device="cuda:0", final_layer=False):
        super(DiffPool, self).__init__()
        self.config = config
        self.device = device
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.embed = GAT(self.feature_size, self.feature_size, 0, self.config['dropout'], alpha=0)
        self.pool = GAT(self.feature_size, self.output_dim, 0, self.config['dropout'], alpha=0)
        self.final_layer = final_layer

    def forward(self, x, a):
        z = self.embed(x, a)
        if self.final_layer:
            s = torch.ones(x.size(0), self.output_dim, device=self.device)
        else:
            s = F.softmax(self.pool(x, a), dim=1)
        x_new = s.transpose(1, 2) @ z
        a_new = s.transpose(1, 2) @ a @ s
        return x_new, a_new


class NGR(BaseModel):
    def __init__(self, config, image_embedding, sentence_embedding, sentence_knowledge, image_knowledge, text_graph,
                 entity_embedding, device=torch.device('cuda:3')):
        super(NGR, self).__init__()
        self.device = device
        self.config = config
        self.hid_size = self.config['node_feature_size']
        self.num_layers = self.config['Graph_learning_layers']  # defult 2, --->[2,3,4]
        # debug--->
        self.image_embedding = image_embedding
        self.sentence_embedding = sentence_embedding
        self.sentence_knowledge = sentence_knowledge
        self.image_knowledge = image_knowledge
        self.text_graph = text_graph
        self.entity_embedding = entity_embedding
        self.max_nodes = 80
        self.max_sentence = 30
        # 定义三个投影头(768)-> 512, (2048)-> 512, (1000)-> 512
        self.text_linear = nn.Linear(768, self.config['node_feature_size']).to(device)  # 文本嵌入投影
        self.image_linear = nn.Linear(2048, self.config['node_feature_size']).to(device)  # 图像嵌入投影
        self.entity_linear = nn.Linear(100, self.config['node_feature_size']).to(device)  # 实体嵌入投影

        self.cross_GAT_layers = GAT(self.config['node_feature_size'], self.config['node_feature_size'], 0,
                                    self.config['dropout'], alpha=0).to(device)

        self.poollayer1 = DiffPool(self.config, self.config['node_feature_size'],
                                   int(self.max_nodes * config['Thresholdpool'])).to(device)
        self.poollayer2 = DiffPool(self.config, self.config['node_feature_size'], 1).to(device)

        self.adjust_layers = nn.ModuleList(
            [GAT(self.config['node_feature_size'], self.config['node_feature_size'], 0, self.config['dropout'], alpha=0)
             for i in range(self.num_layers)]).to(device)
        self.graph_learner = GraphLearner(input_size=self.config['node_feature_size'],
                                          hidden_size=self.config['node_feature_size'],
                                          graph_type=self.config['graph_type'], top_k=self.config['top_k'],
                                          epsilon=self.config['epsilon'], num_pers=self.config['num_per'],
                                          metric_type=self.config['graph_metric_type'],
                                          temperature=self.config['temperature'],
                                          feature_denoise=self.config['feature_denoise'], device=self.device)

        # self.fake_classifier = nn.Sequential(
        #     nn.Linear(self.config['node_feature_size'], self.config['node_feature_size'] * 2),
        #     nn.LayerNorm(self.config['node_feature_size'] * 2),
        #     nn.GELU(),
        #     nn.Linear(self.config['node_feature_size'] * 2, 2),
        #     nn.Sigmoid(),
        # ).to(device)
        # classifier = 1
        self.fake_classifier = nn.Sequential(
            nn.Linear(self.config['node_feature_size'], self.config['node_feature_size'] // 2),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(self.config['node_feature_size'] // 2, 2)).to(device)

        self.fc_mu = nn.Linear(self.hid_size, self.hid_size).to(device)

    def forward(self, adaptive_graph, adaptive_relation, news, attention_mask=None, aux_mask=None):
        # forward
        """使用construct_graph方法构建异构图"""
        node_feature, adj, mask_vector, edge_mask = self.construct_graph(adaptive_graph, adaptive_relation, news)
        hidden_states = self.fc_mu(node_feature)
        """使用enrichment方法对生成的图进行增强
        其中的参数Thresholdsim对应于论文中的epsilon，是相似度计算的截断阈值，用于初步筛选可能相关的边。
        """
        adj = self.enrichment(hidden_states, adj, mask_vector)

        hidden_states = self.cross_GAT_layers(hidden_states, adj)

        for layer in self.adjust_layers:
            new_feature, new_adj = self.learn_graph(node_features=hidden_states,
                                                    graph_skip_conn=self.config['graph_skip_conn'],
                                                    graph_include_self=self.config['graph_include_self'],
                                                    init_adj=adj, node_mask=None)
            adj = torch.mul(new_adj, edge_mask)
            hidden_states = layer(new_feature, adj)

        hidden_states = self.cross_GAT_layers(hidden_states, adj)

        # graph_embedding = torch.max(hidden_states, dim=1)[0]
        # graph_embedding = torch.squeeze(torch.mean(hidden_states, dim=1, keepdim=True), dim=1)
        hidden_states, adj = self.poollayer1(hidden_states, adj)
        hidden_states, adj = self.poollayer2(hidden_states, adj)

        graph_embedding = torch.squeeze(hidden_states, dim=1)
        predict = self.fake_classifier(graph_embedding)
        # predict = torch.squeeze(predict, dim=1)
        return hidden_states, adj, predict

    def _topic_words_attention(self, topic_words_rep, compressed_rep):
        """
            将从潜在主题模型中检索到的主题关键字集成到压缩表示中，以增强上下文。
        """
        _x = compressed_rep.unsqueeze(1).repeat(1, topic_words_rep.size(1), 1)
        attn_weights = torch.sigmoid(torch.sum(_x * topic_words_rep, dim=-1))
        attn_weights = attn_weights.unsqueeze(2).repeat(1, 1, topic_words_rep.size(2))
        out = torch.sum(topic_words_rep * attn_weights, dim=1)
        return out

    def _topic_reconstruction_loss(self, text_inputs, visual_inputs, text_word_dists, visual_word_dists, prior_mean,
                                   prior_variance, posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (
                var_division + diff_term - self.n_components + logvar_det_division)

        # Reconstruction term
        T_RL = -torch.sum(text_inputs * torch.log(text_word_dists + 1e-10), dim=1)

        V_RL = -torch.sum(visual_inputs * torch.log(visual_word_dists + 1e-10), dim=1)

        return KL, T_RL, V_RL

    def learn_graph(self, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None,
                    node_mask=None):
        new_feature, new_adj = self.graph_learner(node_features, node_mask=node_mask)
        bsz = node_features.size(0)
        if graph_skip_conn in (0.0, None):
            # add I
            if graph_include_self:
                if torch.cuda.is_available():
                    new_adj = new_adj + torch.stack([torch.eye(new_adj.size(1)) for _ in range(bsz)], dim=0).to(
                        self.device)
                else:
                    new_adj = new_adj + torch.stack([torch.eye(new_adj.size(1)) for _ in range(bsz)], dim=0).to(
                        self.device)
        else:
            # skip connection
            new_adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * new_adj

        return new_feature, new_adj

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def cnt_edges(self, adj):
        e = torch.ones_like(adj)
        o = torch.zeros_like(adj)
        a = torch.where(adj > 0.0, e, o)
        from torch_geometric.utils import remove_self_loops
        edge_number = remove_self_loops(edge_index=dense_to_sparse(a)[0])[0].size(1) / 2
        return edge_number

    def reset_parameters(self):
        self.text_linear.reset_parameters()
        self.vision_linear.reset_parameters()
        self.GAT_layer.reset_parameters()
        self.graph_learner.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    # 构建异构新闻图（包含三类节点和五种边）
    def construct_graph(self, adaptive_graph, adaptive_relation, news):
        batch_size = adaptive_graph[0].size()[0]
        hop_count = len(adaptive_graph)
        max_nodes = self.max_nodes  # pheme is 30 politifact is 80 gossipcop is 80
        max_sentence = self.max_sentence  # pheme is none politifact is 30 gossipcop is 30

        entity_edge = {}
        for i in range(batch_size):
            entity_edge[i] = {}
            for hop in range(hop_count):
                for entity_id_index in range(len(adaptive_graph[hop][i])):
                    entity_id = adaptive_graph[hop][i][entity_id_index].item()
                    if entity_id != 0 and entity_id not in entity_edge[i]:
                        entity_edge[i][entity_id] = []
                        if hop < hop_count - 1:
                            top_k = len(adaptive_graph[hop + 1][i]) // len(adaptive_graph[hop][i])
                            entity_edge[i][entity_id] = [adaptive_graph[hop + 1][i][entity_id_index * top_k + j].item()
                                                         for j in range(top_k) if adaptive_graph[hop + 1][i][
                                                             entity_id_index * top_k + j].item() != 0]

        knowledge_adj_dic = {}
        knode_to_entityid = {}
        entityid_to_knode = {}
        for i in range(batch_size):
            knode_to_entityid[i] = {}
            entityid_to_knode[i] = {}
            knowledge_adj_dic[i] = [[0 for _ in range(len(entity_edge[i]))] for _ in range(len(entity_edge[i]))]
            index = 0
            for key in entity_edge[i]:
                knode_to_entityid[i][index] = key
                entityid_to_knode[i][key] = index
                index += 1
            for node1, node2_list in entity_edge[i].items():
                for node2 in node2_list:
                    knowledge_adj_dic[i][entityid_to_knode[i][node1]][entityid_to_knode[i][node2]] = 1
                    knowledge_adj_dic[i][entityid_to_knode[i][node2]][entityid_to_knode[i][node1]] = 1

        adj = torch.zeros(batch_size, max_nodes, max_nodes)
        for i in range(batch_size):
            newid = news[i]
            text_adj = self.text_graph[str(newid)]
            sentence_num = len(text_adj)
            if sentence_num > max_sentence:
                sentence_num = max_sentence
                text_adj = [row[:max_sentence] for row in text_adj[:max_sentence]]
            for r in range(len(text_adj)):
                for c in range(len(text_adj[r])):
                    if text_adj[r][c] == 1:
                        adj[i][r][c] = 1
            knowledge_adj = knowledge_adj_dic[i]
            if len(knowledge_adj) > max_nodes - sentence_num - 1:
                knowledge_adj = [row[:max_nodes - sentence_num - 1] for row in
                                 knowledge_adj[:max_nodes - sentence_num - 1]]
            for r in range(len(knowledge_adj)):
                for c in range(len(knowledge_adj[r])):
                    if knowledge_adj[r][c] == 1:
                        adj[i][r + sentence_num + 1][c + sentence_num + 1] = 1
            text_knowledge = self.sentence_knowledge[str(newid)]
            for k, v in text_knowledge.items():
                for entity in v:
                    try:
                        adj[i][int(k)][entityid_to_knode[i][entity] + sentence_num + 1] = 1
                        adj[i][entityid_to_knode[i][entity] + sentence_num + 1][int(k)] = 1
                    except KeyError:
                        hh = 0
                    except IndexError:
                        hh = 0
            image_knowledge = self.image_knowledge[str(newid)]
            for entity in image_knowledge:
                try:
                    adj[i][sentence_num][entityid_to_knode[i][entity] + sentence_num + 1] = 1
                    adj[i][entityid_to_knode[i][entity] + sentence_num + 1][sentence_num] = 1
                except KeyError:
                    hh = 0
                except IndexError:
                    hh = 0

        node_feature = torch.zeros(batch_size, max_nodes, self.hid_size)
        for i in range(batch_size):
            newid = str(news[i])
            sentence_num = len(self.text_graph[str(newid)])
            if sentence_num > max_sentence:
                sentence_num = max_sentence

            for j in range(sentence_num):
                # 这里的sentence_embedding有可能为空，因此需要提前判断，如果为空，则用(768)形状的0填充
                if self.sentence_embedding[newid]:
                    # print(f"Sentence embedding of {newid} is existed\n")
                    node_feature[i][j] = self.text_linear(
                        torch.tensor(self.sentence_embedding[newid][str(j)]).to(self.device))
                else:
                    # print(f"Sentence embedding of {newid} is not existed, use zero vector\n")
                    node_feature[i][j] = torch.zeros(self.hid_size, device=self.device)
            # The expanded size of the tensor (768) must match the existing size (2048) at non-singleton
            # dimension 0.  Target sizes: [768].  Tensor sizes: [2048]

            if newid in self.image_embedding:
                embed = self.image_embedding[newid]
                if not isinstance(embed, torch.Tensor):
                    embed = torch.tensor(embed, device=self.device)
                else:
                    embed = embed.to(self.device)
                node_feature[i][sentence_num] = self.image_linear(embed)
            else:
                node_feature[i][sentence_num] = torch.zeros(self.hid_size, device=self.device)

            for k in knode_to_entityid[i]:
                try:
                    node_feature[i][sentence_num + 1 + k] = self.entity_linear(
                        self.entity_embedding[knode_to_entityid[i][k]].to(self.device))
                except IndexError:
                    continue

        mask_vector = {}
        edge_mask = torch.ones(batch_size, max_nodes, max_nodes)
        for i in range(batch_size):
            newid = str(news[i])
            sentence_num = len(self.text_graph[str(newid)])
            if sentence_num > max_sentence:
                sentence_num = max_sentence
            mask_vector[i] = [j for j in range(sentence_num + 1 + len(entity_edge[i]), max_nodes)]

            edge_mask[i][torch.tensor(mask_vector[i]), :] = 0
            edge_mask[i][:, torch.tensor(mask_vector[i])] = 0

        return node_feature.to(self.device), adj.to(self.device), mask_vector, edge_mask.to(self.device)

    # 对应于论文中的一构图细化部分
    """
        - 基于余弦相似度添加新边（公式5）
        - 使用阈值 Thresholdsim 控制稀疏性---> 对应于论文中的epsilon超参数
        - 通过 cosine_similarity() 计算语义相似性
    """

    def enrichment(self, node_feature, adj, mask_vector):
        for batch in range(adj.size()[0]):
            similarities = self.cosine_similarity(node_feature[batch], node_feature[batch])
            indices = torch.nonzero(similarities > self.config['Thresholdsim']).squeeze()
            adj[batch][indices[:, 0], indices[:, 1]] = 1
            adj[batch][torch.tensor(mask_vector[batch]).to(self.device), :] = 0
            adj[batch][:, torch.tensor(mask_vector[batch]).to(self.device)] = 0

        return adj

    def cosine_similarity(self, v1, v2):
        v1_norm = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2_norm = v2 / torch.norm(v2, dim=1, keepdim=True)
        similarity = torch.matmul(v1_norm, v2_norm.t())
        return similarity

# def read_npy():
#     doc_entity = np.load("./text_graph/doc_entity_dict.npy", allow_pickle=True).item()
#     print("read")
#
