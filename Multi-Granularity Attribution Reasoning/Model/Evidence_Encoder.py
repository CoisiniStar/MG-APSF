"""
获取JSON中分数最高的实体的描述信息作为evidence输入到时空模型中
"""

"""
Evidence Encoder
"""

import json
import os
import re
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm.auto import tqdm
from torch.utils.data import Dataset


def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(?P<url>https?://[^\s]+)', r'', text)
    text = re.sub(r"\@(\w+)", "", text)
    text = text.replace('#', '')
    return text


class JsonDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]
        unique_id = item['Id']

        # 筛选出score > 0.4的实体
        valid_entities = [ent for ent in item['tag_entities'] if ent['score'] > 0.4]

        # 获取最高分的实体描述
        if valid_entities:
            max_score_entity = max(valid_entities, key=lambda x: x['score'])
            description = max_score_entity['description']
        else:
            description = ""

        # 文本预处理
        processed_text = text_preprocessing(description)

        # 对文本进行编码
        encoded_sent = self.tokenizer.encode_plus(
            text=processed_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )

        input_ids = torch.tensor(encoded_sent.get('input_ids'))
        attention_mask = torch.tensor(encoded_sent.get('attention_mask'))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'unique_id': unique_id,
            'processed_text': processed_text
        }


def process_and_store(bert, device, dataset, store_dir):
    os.makedirs(store_dir, exist_ok=True)
    bert.eval()

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        unique_id = sample['unique_id']
        processed_text = sample['processed_text']

        # 处理空描述的情况
        if processed_text == "":
            # 创建全零的token级别表示 [1, 768]
            token_emb = np.zeros((1, 768), dtype=np.float32)
            # 创建全零的CLS级别表示 [768]
            cls_emb = np.zeros(768, dtype=np.float32)
        else:
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = bert(input_ids=input_ids, attention_mask=attention_mask)

            # 获取token级别的表示（排除[CLS]和[SEP]）
            token_emb = outputs.last_hidden_state[:, 1:-1, :].detach().cpu().squeeze(0).numpy()
            # 获取CLS级别的表示
            cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().squeeze(0).numpy()

        # 保存token级别的嵌入
        token_filename = os.path.join(store_dir, f"{unique_id}evidence.npy")
        np.save(token_filename, token_emb)

        # 保存CLS级别的嵌入
        cls_filename = os.path.join(store_dir, f"{unique_id}evidence_full_text.npy")
        np.save(cls_filename, cls_emb)


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 配置文件路径
    json_path = "output_QID_dataset.json"
    store_dir = "./Evidence_Embedding_File/"
    model_path = "./bert/"

    # 加载JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from JSON file")

    # 加载BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained(model_path).to(device)
    print("Loaded BERT model and tokenizer")

    # 创建数据集
    dataset = JsonDataset(data, tokenizer)
    print("Created dataset")

    # 处理并存储嵌入
    process_and_store(bert_model, device, dataset, store_dir)
    print(f"Embeddings saved to {store_dir}")