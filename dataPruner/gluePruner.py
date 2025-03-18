# -- coding: utf-8 --
from torch.utils.data.dataloader import DataLoader
import numpy as np
from transformers.data.data_collator import DataCollatorWithPadding
import torch
from transformers import BertTokenizer,AutoTokenizer
from torch.utils.data import SubsetRandomSampler
from datasets import Dataset


class GLUEPruner():
    def __init__(self, dataset, ratio=0.5):
        self.dataset = dataset
        self.keep_ratio = 1 - ratio if ratio <= 1 else (100 - ratio) / 100
        self.index = range(len(self.dataset))
        self.device = torch.device('cuda:0')
        self.num_updated_samples = 0
        self.scores = torch.zeros([len(self.dataset)]).cpu()
        self.cur_batch_index = None
        self.iteration = 0
        self.cur_index = None
        self.num_pruned_samples = 0

    def update(self, values, index):
        self.cur_batch_index = index
        s = torch.abs(values.to(dtype=self.scores.dtype).cpu())
        # 更新分数，对传进来的样本对应的索引更新
        # 获取更新的样本个数
        self.num_updated_samples = self.num_updated_samples + len(values)

        # 打印更新的样本个数

        self.scores[self.cur_batch_index.cpu()] = s
        # self.scores[self.cur_batch_index.cpu()] = s

    def update_keep_ratio(self, total_iteration):
        keep_ratio = 1 - self.iteration / total_iteration
        if keep_ratio > 0.1:
            self.keep_ratio = keep_ratio

    def prune(self):
        if self.iteration == 0:
            remain_indices = np.arange(len(self.dataset))
            np.random.shuffle(remain_indices)
            self.cur_index = remain_indices
            self.iteration += 1
            return

        def fraction_threshold(tensor, fraction):
            # 返回阈值，例如我从传进来的分数中，选取前百分之90大的，返回这些最大值中的最小值
            threshold, _ = torch.topk(tensor, int((fraction) * len(tensor)))

            return threshold[-1]

        def threshold_mask(tensor, threshold):
            assert isinstance(tensor, torch.Tensor)
            idx = tensor < threshold
            mask = torch.ones_like(tensor, device=torch.device('cuda:0'))
            mask[idx] = 0
            return mask

        print(self.keep_ratio)
        threshold = fraction_threshold(self.scores, self.keep_ratio)

        remain_mask = threshold_mask(self.scores, threshold).cpu().numpy().astype(bool)

        remain_indices = np.where(remain_mask)[0]
        print('remain_samples', len(remain_indices))
        np.random.shuffle(remain_indices)
        self.cur_index = remain_indices
        self.iteration += 1

    def prune1(self, history_len):
        if self.iteration == 0:
            remain_indices = np.arange(len(self.dataset))
            np.random.shuffle(remain_indices)
            self.cur_index = remain_indices
            self.iteration += 1
            return

        def fraction_threshold(tensor, fraction, history_len):
            # 返回阈值，例如我从传进来的分数中，选取前百分之90大的，返回这些最大值中的最小值
            threshold, _ = torch.topk(tensor, int((fraction) * history_len))
            print(f'保留{int((fraction) * history_len)}条数据')

            return threshold[-1]

        def select_topk_indices(tensor, fraction, history_len):
            num_to_keep = int(fraction * history_len)
            #             print(f"保留{num_to_keep}条数据")
            _, topk_indices = torch.topk(tensor, num_to_keep)
            return topk_indices.cpu().numpy()

        def threshold_mask(tensor, threshold, size):
            assert isinstance(tensor, torch.Tensor)
            print("Tensor 的形状:", tensor.shape)
            zeros_count = (tensor == 0).sum().item()

            # 统计不为 0 的元素个数
            nonzeros_count = (tensor != 0).sum().item()

            print(f"Tensor 中值为 0 的元素个数: {zeros_count}")
            print(f"Tensor 中不为 0 的元素个数: {nonzeros_count}")
            print(f'阈值为{threshold}')
            idx = tensor < threshold
            print(f'idx的大小为{len(idx)}')
            #             mask = torch.ones(size, device=torch.device('cuda:0'))

            mask = torch.ones_like(tensor, device=torch.device('cuda:0'))
            mask[idx] = 0
            print("mask中为1的个数有")
            print(torch.sum(mask).item())
            return mask

        print(self.keep_ratio)
        #         threshold = fraction_threshold(self.scores, self.keep_ratio,history_len)

        #         remain_mask = threshold_mask(self.scores, threshold,history_len).cpu().numpy().astype(bool)
        remain_indices = select_topk_indices(self.scores, self.keep_ratio, history_len)

        #       remain_indices = np.where(remain_mask)[0]
        print('remain_samples', len(remain_indices))
        np.random.shuffle(remain_indices)
        self.cur_index = remain_indices
        self.iteration += 1
        self.scores = torch.zeros([len(self.dataset)]).cpu()
        #         print(f"更新的样本个数: {self.num_updated_samples}")
        self.num_updated_samples = 0

    def pro_loss(self, loss):
        loss *= 1 / self.keep_ratio
        return loss

    def get_sampler(self):
        sampler = SubsetRandomSampler(self.cur_index)
        return sampler

    @property
    def save_ratio(self):
        return (self.num_pruned_samples) / (self.iteration * len(self.dataset))

    def get_pruned_train_dataset(self):
        traindataset = {k: [v[i] for i in self.cur_index] for k, v in self.dataset.items()}
        train_dataset = Dataset.from_dict(traindataset)
        return train_dataset


def get_pruned_dataloader(config, dataset, sampler):
    model_checkpoint = "bert-base-uncased"
    batch_size = config.batchsize
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=True,
        num_workers=0,
        pin_memory=True, sampler=sampler
    )
    return train_dataloader


def get_pruned_dataloader_1(config, dataset, sampler):
    model_checkpoint = "bert-base-uncased"
    batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=data_collator,
        # drop_last=True,
        num_workers=0,
        pin_memory=True, sampler=sampler
    )
    return train_dataloader


from transformers import default_data_collator


def get_squad_dataloader(config, dataset, sampler):
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.batchsize,
    )
    return train_dataloader






