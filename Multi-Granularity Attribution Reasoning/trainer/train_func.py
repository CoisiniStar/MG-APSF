import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup
def safe_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, labels=[0, 1])

    # 手动添加准确率
    report['accuracy'] = accuracy_score(y_true, y_pred)

    # 处理单类别情况
    if len(np.unique(y_true)) == 1:
        print("警告：数据集中只有单一类别")
        # 添加额外指标
        majority_class = np.unique(y_true)[0]
        report['majority_class_accuracy'] = np.mean(y_pred == majority_class)

    return report

# rewards_steps是一个list，其中有三个都是tensor的元素：tensor(32,5), tensor(32,15), tensor(32,30)
# config['alpha2'] = 0.1;
def actor_critic_loss(config, rewards_steps, act_probs_steps, state_values_steps, classifier_loss, actor_loss_list, critic_loss_list):
    num_steps = len(rewards_steps)
    for i in range(num_steps):
        shape0 = state_values_steps[i].shape[0]
        shape1 = state_values_steps[i].shape[1]
        shape2 = config['topk'][i] # baseline: (128,15,1)
        baseline = state_values_steps[i] if i>=1 else state_values_steps[i][:,:,None]
        if i==num_steps-1:  # i=2 时,
            terminal_reward = config['alpha1'] * (1 - classifier_loss.detach())  # rewards_steps(list:3->tensor(128,5),tensor(128,15),tensor(128,30));
            td_error = config['alpha2'] * rewards_steps[i] + ((1-config['alpha2']) * terminal_reward)[:,None] - baseline.expand(shape0, shape1, shape2).reshape(shape0, shape1*shape2)
        else:
            td_error = config['alpha2'] * rewards_steps[i] + config['gamma'] * state_values_steps[i+1].squeeze(-1) - baseline.expand(shape0, shape1, shape2).reshape(shape0, shape1*shape2)
        actor_loss = - torch.log(act_probs_steps[i]) * td_error.detach()
        critic_loss = td_error.pow(2)
        actor_loss_list.append(actor_loss.mean())
        critic_loss_list.append(critic_loss.mean())


def train_func(config, epoch, model, dataloader, device, optimizer, scheduler, loss_weight, num_train_steps):
    model.train()
    anchor_all_loss = 0.0
    critic_all_loss = 0.0
    actor_all_loss = 0.0
    targets = []
    predictions = []
    total_loss = 0.0
    optimizer_anchor = torch.optim.AdamW(model.AKAN.parameters(), lr=config['lr_anchor'])
    scheduler_anchor = get_linear_schedule_with_warmup(
        optimizer=optimizer_anchor,
        num_warmup_steps=5,
        num_training_steps=num_train_steps
    )

    entityloss = nn.BCELoss()  # anchor
    spaceloss = nn.BCELoss()
    eventloss = nn.BCELoss()

    with tqdm(dataloader, unit="batch", total=len(dataloader)) as single_epoch:

        for step, batch in enumerate(single_epoch):
            # print(f"Batch {step} type: {type(batch)}, length: {len(batch)}")
            img_list = []
            text_list = []
            event_list = []
            evidence_list = []
            label_list = []
            news_list = []
            time_list = []
            single_epoch.set_description(f"Training- Epoch {epoch}")
            for item in batch:
                img_list.append(item[0])
                text_list.append(item[1])
                event_list.append(item[2])
                evidence_list.append(item[3])
                label_list.append(item[4])
                news_list.append(int(item[5]['Id'].strip()))  # news_id
                time_list.append(item[5]['create_time'])  # create_time

            batch_data = (
                torch.stack(img_list, dim=0),  # 形状变为 (128, 28, 768)
                torch.stack(text_list, dim=0),  # 形状变为 (128, 512, 768)
                torch.stack(event_list, dim=0),  # 形状变为 (128, 129, 768)
                torch.stack(evidence_list, dim=0),  # 形状变为 (128, 2, 768)
                torch.tensor(label_list),  # 形状变为 (128,)
                news_list,
                torch.tensor(time_list)  # 形状变为 (128,)
            )
            batched_img, batched_text, batched_event, batched_evidence, batched_labels, batched_news, batched_time = batch_data
            batched_labels = batched_labels.to(device)                            # news_list, batched_img, batched_text, batched_event, batched_evidence, batched_labels
            batched_logits, score_anchor_graph, score_gnn_model, score_space_model, rewards_steps1, act_probs_steps1, state_values_steps1 = model(
                                                                                                                                    news=batched_news,
                                                                                                                                    batched_img=batched_img,
                                                                                                                                    batched_text=batched_text,
                                                                                                                                    batched_event=batched_event,
                                                                                                                                    batched_evidence=batched_evidence,
                                                                                                                                    batched_time_info=batched_time,
                                                                                                                                         batched_labels=batched_labels)

            # 将模型输出的所有变量移动到device上
            batched_logits = batched_logits.to(device)
            score_anchor_graph = score_anchor_graph.to(device)
            score_gnn_model = score_gnn_model.to(device)
            score_space_model = score_space_model.to(device)
            # 处理rewards_steps1, act_probs_steps1和state_values_steps1列表中的每个tensor
            rewards_steps1 = [r.to(device) if isinstance(r, torch.Tensor) else r for r in rewards_steps1]
            act_probs_steps1 = [a.to(device) if isinstance(a, torch.Tensor) else a for a in act_probs_steps1]
            state_values_steps1 = [s.to(device) if isinstance(s, torch.Tensor) else s for s in state_values_steps1]

            _, pred = torch.max(batched_logits, dim=-1)

            batched_labels = batched_labels.float()
            predictions.append(pred.cpu().numpy())
            targets.append(batched_labels.cpu().numpy())

            entity_loss = entityloss(score_anchor_graph.squeeze(-1), batched_labels)
            event_loss = eventloss(score_gnn_model.squeeze(-1), batched_labels)
            space_loss = spaceloss(score_space_model.squeeze(-1), batched_labels)
            loss = F.cross_entropy(batched_logits, batched_labels.long(), weight=loss_weight.to(device), reduction='none')
            all_loss = loss + 0.2 * (entity_loss + event_loss + space_loss)
            total_loss += all_loss.mean().item()
            ##### 强化学习 #####
            actor_loss_list = []
            critic_loss_list = []
            actor_critic_loss(config, rewards_steps1, act_probs_steps1, state_values_steps1, loss.detach(), actor_loss_list, critic_loss_list)
            actor_losses = torch.stack(actor_loss_list).sum()
            critic_losses = torch.stack(critic_loss_list).sum()
            anchor_loss = actor_losses + critic_losses

            all_loss.mean().backward(retain_graph=True)

            if step % config["gradient_accumulation_steps"] == 0 or step == len(dataloader) - 1:
                # 只对非AKAN参数进行梯度裁剪
                non_akan_params = [param for name, param in model.named_parameters()
                                   if not name.startswith('AKAN.') and param.requires_grad and param.grad is not None]
                if non_akan_params:
                    torch.nn.utils.clip_grad_norm_(non_akan_params, 0.5)

                optimizer.step()
                scheduler.step()
                # optimizer.zero_grad()  # 清空主模型梯度
                # 第二步：更新AKAN参数
            anchor_loss.backward()

            if step % config["gradient_accumulation_steps"] == 0 or step == len(dataloader) - 1:
                optimizer_anchor.step()
                scheduler_anchor.step()

            # ===== 关键修改：手动清空AKAN梯度 =====
            model.AKAN.zero_grad()
            # ===== 清空主模型梯度 =====
            optimizer.zero_grad()
            critic_all_loss = critic_all_loss + critic_losses.item()
            actor_all_loss = actor_all_loss + actor_losses.item()
            anchor_all_loss = anchor_all_loss + anchor_loss.item()
            single_epoch.set_postfix(train_loss=total_loss / (step + 1))

    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    report = safe_classification_report(targets, predictions)
    epoch_train_loss = total_loss / len(dataloader)
    return epoch_train_loss, report



def eval_func(model, dataloader, device):
    model.eval()
    total_loss = 0
    targets = []
    predictions = []

    with tqdm(dataloader, unit="batch", total=len(dataloader)) as single_epoch:

        for step, batch in enumerate(single_epoch):
            # print(f"Batch {step} type: {type(batch)}, length: {len(batch)}")
            img_list = []
            text_list = []
            event_list = []
            evidence_list = []
            label_list = []
            news_list = []
            time_list = []
            for item in batch:
                img_list.append(item[0])
                text_list.append(item[1])
                event_list.append(item[2])
                evidence_list.append(item[3])
                label_list.append(item[4])
                news_list.append(int(item[5]['Id'].strip()))  # news_id
                time_list.append(item[5]['create_time'])  # create_time

            batch_data = (
                torch.stack(img_list, dim=0),  # 形状变为 (128, 28, 768)
                torch.stack(text_list, dim=0),  # 形状变为 (128, 512, 768)
                torch.stack(event_list, dim=0),  # 形状变为 (128, 129, 768)
                torch.stack(evidence_list, dim=0),  # 形状变为 (128, 2, 768)
                torch.tensor(label_list),  # 形状变为 (128,)
                news_list,
                torch.tensor(time_list)  # 形状变为 (128,)
            )
            batched_img, batched_text, batched_event, batched_evidence, batched_labels, batched_news, batched_time = batch_data
            batched_labels = batched_labels.to(device)                            # news_list, batched_img, batched_text, batched_event, batched_evidence, batched_labels
            with torch.no_grad():
                batched_logits, score_anchor_graph, score_gnn_model, score_space_model, rewards_steps1, act_probs_steps1, state_values_steps1 = model(
                                                                                                                                    news=batched_news,
                                                                                                                                    batched_img=batched_img,
                                                                                                                                    batched_text=batched_text,
                                                                                                                                    batched_event=batched_event,
                                                                                                                                    batched_evidence=batched_evidence,
                                                                                                                                    batched_time_info=batched_time,
                                                                                                                                         batched_labels=batched_labels)

            batched_logits = batched_logits.to(device)

            _, pred = torch.max(batched_logits, dim=-1)

            batched_labels = batched_labels.float()
            predictions.append(pred.cpu().numpy())
            targets.append(batched_labels.cpu().numpy())


    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    epoch_validation_loss = total_loss / len(dataloader)
    report = safe_classification_report(targets, predictions)

    acc = report['accuracy']
    f1_score = report['weighted avg']['f1-score']
    prec = report['weighted avg']['precision']
    rec = report['weighted avg']['recall']
    return epoch_validation_loss, report, acc, prec, rec, f1_score
