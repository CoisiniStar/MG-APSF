## Importing libraries
import numpy as np
import pandas as pd
import math
import ctypes
import random

lib_path = "/home/AnonymousUser/cuda12.1/lib64/libcusparse.so.12"
ctypes.CDLL(lib_path)
from torch.utils.data import Sampler
from Model.MGFramework import Detection_Module
import dgl
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import conv

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import config
from transformers import get_linear_schedule_with_warmup
from utils.util import seed_everything, prepare_device
from Combinated_Dataset_AMG import *
# import config, Event_GModel, dataset, engine, utils
import numpy as np

from trainer.train_func import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 自定义collate_fn函数
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


# test---module
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnchorKG')

    parser.add_argument('-c', '--config', default="./config/anchorkg_config.json", type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default="cuda:5", type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--use_nni', action='store_true', help='use nni to tune hyperparameters')

    config = ConfigParser.from_args(parser)

    device = config["device"]

    loss_weight, dataset_train, dataset_val, dataset_test = load_and_shuffle_dataset(config)
    entity_id_dict, relation_adj, entity_adj, entity_embedding, relation_embedding, doc_entity_dict, entity_doc_dict, neibor_embedding, neibor_num, image_embedding, sentence_embedding, sentence_knowledge, image_knowledge, text_graph = load_npy_file(
        config)
    ## Setup the dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config["batch_size"],
        drop_last=False,
        shuffle=True,
        collate_fn=collate_fn
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config["batch_size"],
        drop_last=False,
        shuffle=True,
        collate_fn=collate_fn
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config["batch_size"],
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn
    )
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config["gradient_accumulation_steps"])
    num_train_steps = num_update_steps_per_epoch * config["epochs"]


    classifier = Detection_Module(config, doc_entity_dict, entity_doc_dict, config['amg_text_vec_dir'], entity_adj,
                                  relation_adj, entity_id_dict, neibor_embedding,
                                  neibor_num, entity_embedding, relation_embedding, device=config["device"],
                                  image_embedding=image_embedding, sentence_embedding=sentence_embedding,
                                  sentence_knowledge=sentence_knowledge, image_knowledge=image_knowledge,
                                  text_graph=text_graph, in_feats_embedding=[768, 512], out_feats_embedding=[512, 256],
                                  classifier_dims=[128], dropout_p=0.6, n_classes=2)


    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 ** 2
        return size_mb


    # 使用方法
    print(f"模型参数占用显存: {get_model_size(classifier):.2f} MB")
    # if torch.cuda.is_available():
    #     parallel_model = torch.nn.DataParallel(classifier, device_ids=[1,2,3])
    #     parallel_model = parallel_model.cuda()
    # classifier.to(config["device"])


    optimizer = AdamW(classifier.parameters(), lr=config["lr"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    best_loss = np.inf
    best_val_f1 = 0
    best_epoch = 0
    epochs = config["epochs"]
    for epoch in range(epochs):
        print(f"\n{'=' * 30} Epoch {epoch + 1}/{epochs} {'=' * 30}")
        ## 训练阶段
        train_loss, train_report = train_func(
            config,
            epoch + 1,
            classifier,
            dataloader_train,
            device,
            optimizer,
            scheduler,
            loss_weight=loss_weight,
            num_train_steps=num_train_steps
        )
        ## 验证阶段 (使用验证集而非测试集)
        val_loss, val_report, val_acc, val_prec, val_rec, val_f1 = eval_func(
            classifier,
            dataloader_val,
            device
        )

        # 打印训练和验证结果
        print(f"\nEpoch: {epoch + 1} | Training loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        print("\nTrain Report:")
        print(train_report)
        print("\nValidation Report:")
        print(val_report)
        print(f"Validation Metrics - Accuracy: {val_acc:.6f} | Weighted Precision: {val_prec:.6f} | "
              f"Weighted Recall: {val_rec:.6f} | Weighted F1-score: {val_f1:.6f}")


        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            # 保存模型
            torch.save(classifier.state_dict(), f'best_model.pt')
            print(f"Saved best model at epoch {epoch + 1} with Val F1: {val_f1:.6f}")

        print(f"\n{'=' * 70}")

    # ===== 最终测试阶段 =====
    print(f"\n{'=' * 30} Final Testing {'=' * 30}")
    print(f"Loading best model from epoch {best_epoch} with Val F1: {best_val_f1:.6f}")

    weights = torch.load('best_model.pt', map_location='cpu')
    classifier.load_state_dict(weights)


    # # 在测试集上评估
    test_loss, test_report, test_acc, test_prec, test_rec, test_f1 = eval_func(
        classifier,
        dataloader_test,  # 使用真正的测试集
        device
    )
    # 打印测试结果
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.6f}")
    print(test_report)
    print(f"Test Metrics - Accuracy: {test_acc:.6f} | Weighted Precision: {test_prec:.6f} | "
          f"Weighted Recall: {test_rec:.6f} | Weighted F1-score: {test_f1:.6f}")




