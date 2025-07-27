import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
import random

from parse_config import ConfigParser
import pandas as pd
import argparse

def map_labels(labels):
    """
    将字符串标签列表转为二分类标签（0/1），返回LongTensor
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    labels = [int(l) for l in labels]
    labels = torch.tensor(labels)
    mapped = (labels > 0).long()  # 大于0的映射为1，否则为0
    return mapped


class MGDataset(Dataset):
    def __init__(self, df, root_dir, image_id, text_id,
                 image_vec_dir, text_vec_dir, event_vec_dir, evidence_vec_dir,dataset_name="MGDataset"):
        super(MGDataset, self).__init__()
        self.df = df
        self.root_dir = root_dir
        self.image_id = image_id
        self.text_id = text_id
        self.image_vec_dir = image_vec_dir
        self.text_vec_dir = text_vec_dir
        self.event_vec_dir = event_vec_dir
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(768)
        self.evidence_vec_dir = evidence_vec_dir
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name_image = self.df[self.image_id][idx]
        file_name_text = self.df[self.text_id][idx]
        file_name_event = self.df[self.text_id][idx]  # Assuming this is intentional
        file_name_evidence = self.df[self.text_id][idx]  # Assuming this is intentional

        # Helper function to load or create zero array
        def safe_load_npy(path, default_shape, dtype=np.float32):
            if os.path.exists(path):
                return np.load(path)
            else:
                return np.zeros(default_shape, dtype=dtype)

        # Load or create image vectors
        image_full_path = f'{self.root_dir}{self.image_vec_dir}{file_name_image}_full_image.npy'
        image_vec_path = f'{self.root_dir}{self.image_vec_dir}{file_name_image}.npy'

        image_vec_full = safe_load_npy(image_full_path, (1, 2048))
        image_vec = safe_load_npy(image_vec_path, (1, 2048))
        all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)

        all_image_vec = self.adaptive_pooling(torch.tensor(all_image_vec).float().unsqueeze(0)).squeeze(0)


        # Load or create text vectors
        text_full_path = f'{self.root_dir}{self.text_vec_dir}{file_name_text}_full_text.npy'
        text_vec_path = f'{self.root_dir}{self.text_vec_dir}{file_name_text}.npy'

        text_vec_full = safe_load_npy(text_full_path, (768,))
        text_vec = safe_load_npy(text_vec_path, (1, 768))

        # Ensure proper shapes
        text_vec_full = text_vec_full.reshape(1, -1) if text_vec_full.ndim == 1 else text_vec_full
        text_vec = text_vec.reshape(-1, 768) if text_vec.ndim == 1 else text_vec
        all_text_vec = np.concatenate([text_vec_full, text_vec], axis=0)
        all_text_vec = torch.from_numpy(all_text_vec)

        # Load or create event vectors
        event_tokens_path = f'{self.root_dir}{self.event_vec_dir}{file_name_event}_tokens.npy'
        event_cls_path = f'{self.root_dir}{self.event_vec_dir}{file_name_event}_cls.npy'

        event_vec = safe_load_npy(event_tokens_path, (1, 768))
        event_vec_full = safe_load_npy(event_cls_path, (768,))

        # Ensure proper shapes
        event_vec_full = event_vec_full.reshape(1, -1) if event_vec_full.ndim == 1 else event_vec_full
        event_vec = event_vec.reshape(-1, 768) if event_vec.ndim == 1 else event_vec
        all_event_vec = np.concatenate([event_vec_full, event_vec], axis=0)

        all_event_vec = torch.from_numpy(all_event_vec)

        # Get evidence vectors
        evidence_tokens_path = f'{self.evidence_vec_dir}{file_name_evidence}evidence.npy'
        evidence_cls_path = f'{self.evidence_vec_dir}{file_name_evidence}evidence_full_text.npy'

        evidence_vec = safe_load_npy(evidence_tokens_path, (1, 768))
        evidence_vec_full = safe_load_npy(evidence_cls_path, (768,))

        # Ensure proper shapes
        evidence_vec_full = evidence_vec_full.reshape(1, -1) if evidence_vec_full.ndim == 1 else evidence_vec_full
        evidence_vec = evidence_vec.reshape(-1, 768) if evidence_vec.ndim == 1 else evidence_vec
        all_evidence_vec = np.concatenate([evidence_vec_full, evidence_vec], axis=0)
        all_evidence_vec = torch.from_numpy(all_evidence_vec)


        # Get label
        # label = 1 if self.df['label'][idx] == '1' else 0

        sample = {
            'Id': self.df['unique_id'][idx],
            # 'content': self.df['content'][idx],
            'create_time': self.df['create_time'][idx],
            'label': map_labels(self.df['label'])[idx].clone().detach()
        }

        return all_image_vec, all_text_vec, all_event_vec, all_evidence_vec, sample['label'], sample

def create_dataframe(data_list):
    """从数据列表创建DataFrame"""
    data = []
    for item in data_list:
        data.append({
            'unique_id': item['Id'],
            'label': item['label'],
            'create_time': item.get('create_time', 0)  # 处理缺失的时间戳
        })
    return pd.DataFrame(data)

def load_and_shuffle_dataset(config):
    root_dir = config["root_dir"]
    # 加载所有JSON数据
    def load_json(file_path):
        with open(f"{root_dir}{file_path}", 'r') as f:
            return json.load(f)

    train_data = load_json(config["amg_train_json_name"])
    test_data = load_json(config["amg_test_json_name"])
    val_data = load_json(config["amg_val_json_name"])
    random.shuffle(train_data)
    random.shuffle(test_data)
    random.shuffle(val_data)
    batch_size = config['batch_size']

    tr_labels = [item['label'] for item in train_data]  # 训练集中的标签列表
    tr_mapped_labels = map_labels(tr_labels)
    binary_labels = tr_mapped_labels.clone().detach().long()  # tensor, 元素为0或1
    num_classes = 2
    class_counts = torch.bincount(binary_labels, minlength=num_classes)
    weight = class_counts.float()
    weight = weight.sum() / (num_classes * weight)
    # 打印各类别数量和权重
    print("标签分布：", class_counts.tolist())
    print("类别权重：", weight.tolist())

    # 创建DataFrame
    df_train = create_dataframe(train_data)
    df_val = create_dataframe(val_data)
    df_test = create_dataframe(test_data)

    # 打印数据集大小
    print(f"Train size: {len(df_train)}")
    print(f"Validation size: {len(df_val)}")
    print(f"Test size: {len(df_test)}")

    dataset_train = MGDataset(df_train, config["root_dir"], "unique_id", "unique_id", config["amg_image_vec_dir"],
                                 config["amg_text_vec_dir"], config["amg_event_vec_dir"], config["amg_evidence_vec_dir"])
    dataset_val = MGDataset(df_val, config["root_dir"], "unique_id", "unique_id", config["amg_image_vec_dir"],
                               config["amg_text_vec_dir"], config["amg_event_vec_dir"], config["amg_evidence_vec_dir"])
    dataset_test = MGDataset(df_test, config["root_dir"], "unique_id", "unique_id", config["amg_image_vec_dir"],
                                config["amg_text_vec_dir"], config["amg_event_vec_dir"], config["amg_evidence_vec_dir"])

    return weight, dataset_train, dataset_val, dataset_test


def collate_fn(batch):
    # 修改代码
    # 找到tensor1和tensor2的第一个维度的最大长度
    max_length_dim1 = max(item[0].shape[0] for item in batch)
    max_length_dim2 = max(item[1].shape[0] for item in batch)
    max_length_dim3 = max(item[2].shape[0] for item in batch)
    max_length_dim4 = max(item[3].shape[0] for item in batch)

    # 扩展tensor1和tensor2的第一个维度
    expanded_data = [
        (torch.cat([item[0], torch.zeros(max_length_dim1 - item[0].shape[0], item[0].shape[1])]),
         torch.cat([item[1], torch.zeros(max_length_dim2 - item[1].shape[0], item[1].shape[1])]),
         torch.cat([item[2], torch.zeros(max_length_dim3 - item[2].shape[0], item[2].shape[1])]),
         torch.cat([item[3], torch.zeros(max_length_dim4 - item[3].shape[0], item[3].shape[1])]),
         item[4],
         item[5]
         )
        for item in batch
    ]
    return expanded_data


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='AnchorKG')
#
#     parser.add_argument('-c', '--config', default="./config/anchorkg_config.json", type=str,
#                       help='config file path (default: None)')
#     parser.add_argument('-r', '--resume', default=None, type=str,
#                       help='path to latest checkpoint (default: None)')
#     parser.add_argument('-d', '--device', default=None, type=str,
#                       help='indices of GPUs to enable (default: all)')
#     parser.add_argument('--use_nni', action='store_true', help='use nni to tune hyperparameters')
#
#     config = ConfigParser.from_args(parser)
#     # 加载数据集
#     weight, dataset_train, dataset_val, dataset_test = load_and_shuffle_dataset(config)
#     print(f"Train dataset size: {len(dataset_train)}")
#     print(f"Validation dataset size: {len(dataset_val)}")
#     print(f"Test dataset size: {len(dataset_test)}")
#     print("Weight for loss function:", weight)
#     train_dataloader = DataLoader(
#         dataset_train,
#         batch_size=config["batch_size"],
#         drop_last=False,
#         shuffle=False,
#         collate_fn=collate_fn
#     )
#
#     # 示例代码：迭代 train_dataloader 并打印数据
#     for batch_idx, batch in enumerate(train_dataloader):
#         print(f"Batch {batch_idx} type: {type(batch)}, length: {len(batch)}")
#         img_list = []
#         text_list = []
#         event_list = []
#         evidence_list = []
#         label_list = []
#         news_list = []
#         time_list = []
#         for item in batch:
#             img_list.append(item[0])
#             text_list.append(item[1])
#             event_list.append(item[2])
#             evidence_list.append(item[3])
#             label_list.append(item[4])
#             news_list.append(int(item[5]['Id'].strip()))  # news_id
#             time_list.append(item[5]['create_time'])  # create_time
#         batch_data = (
#             torch.stack(img_list, dim=0),  # 形状变为 (128, 28, 768)
#             torch.stack(text_list, dim=0),  # 形状变为 (128, 512, 768)
#             torch.stack(event_list, dim=0),  # 形状变为 (128, 129, 768)
#             torch.tensor(label_list),  # 形状变为 (128,)
#             news_list,
#             torch.tensor(time_list)  # 形状变为 (128,)
#         )
#         print('finished batch_data\n')





def load_npy_file(config):
    if os.path.exists(config['cache_path']):
        print("Loading data from cache...")
        entity_id_dict = np.load(config['cache_path'] + "/entity_id_dict.npy", allow_pickle=True).item()
        # relation_id_dict = np.load(config['cache_path']+"/relation_id_dict.npy", allow_pickle=True).item()
        entity_adj = torch.load(config['cache_path'] + "/entity_adj.pt")
        relation_adj = torch.load(config['cache_path'] + "/relation_adj.pt")
        entity_embedding = torch.load(config['cache_path'] + "/entity_embedding.pt")
        relation_embedding = torch.load(config['cache_path'] + "/relation_embedding.pt")
        doc_entity_dict = np.load(config['cache_path'] + "/doc_entity_dict.npy", allow_pickle=True).item()
        entity_doc_dict = np.load(config['cache_path'] + "/entity_doc_dict.npy", allow_pickle=True).item()
        # neibor_data = torch.load(config['cache_path'] +"/neibor_data.pt")
        neibor_embedding = torch.load(config['cache_path'] + "/neibor_embedding.pt")
        neibor_num = torch.load(config['cache_path'] + "/neibor_num.pt")
        # The input of the model graph
        # image_embedding = np.load(config['cache_path'] + "/image_feature_embedding.npy")
        # text_embedding = np.load(config['cache_path'] + "/doc_feature_embedding.npy")
        image_embedding = np.load(config['cache_path'] + "/image_feature_embedding.npy", allow_pickle=True).item()
        sentence_embedding = torch.load(config['cache_path'] + "/Seg_Text_Embedding_File/sentence_embeddings_full.pt")

        with open(f"{config['cache_path']}/train_knowledge.json", 'r', encoding='utf-8') as f:
            sentence_knowledge = json.load(f)
        # print("---finished---\n")
        image_knowledge = np.load(config['cache_path'] + "/image_entity_dict.npy", allow_pickle=True).item()
        text_graph = torch.load(config['cache_path'] + "/text_graph.pt")
        # print(len(text_graph))

    else:
        raise FileNotFoundError("The required cache_path does not exist in the specified configuration.")

    return entity_id_dict, relation_adj, entity_adj, entity_embedding, relation_embedding, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, image_embedding, sentence_embedding, sentence_knowledge, image_knowledge, text_graph

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='AnchorKG')
#
#     parser.add_argument('-c', '--config', default="./config/anchorkg_config.json", type=str,
#                       help='config file path (default: None)')
#     parser.add_argument('-r', '--resume', default=None, type=str,
#                       help='path to latest checkpoint (default: None)')
#     parser.add_argument('-d', '--device', default=None, type=str,
#                       help='indices of GPUs to enable (default: all)')
#     parser.add_argument('--use_nni', action='store_true', help='use nni to tune hyperparameters')
#
#     config = ConfigParser.from_args(parser)
#     entity_id_dict, relation_adj, entity_adj, entity_embedding, relation_embedding, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, image_embedding, sentence_embedding, sentence_knowledge, image_knowledge, text_graph = load_npy_file(config)
#     print("Finished loading npy files.")