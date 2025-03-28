from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import RobertaModel, RobertaTokenizer
from transformers import BertTokenizer,BertTokenizerFast,AutoTokenizer
import torch
import os

def get_dataloader(task:str, model_checkpoint:str,dataloader_drop_last:bool=True, shuffle:bool=True,
                   batch_size:int=16, dataloader_num_workers:int=2, dataloader_pin_memory:bool=True,tokenizer=None,only_train=False):

    task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    validation_name = 'validation'

    if task == "mnli":
        validation_name = "validation_matched"
    if task == "mnli-mm":
        validation_name = "validation_mismatched"

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    train_dataset=dataset['train']
    validation_dataset=dataset[validation_name]
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    validation_dataset = validation_dataset.map(preprocess_function, batched=True)

    if 'roberta' in model_checkpoint:
        columns_to_return = ['label', 'idx', 'input_ids', 'attention_mask']
    else:
        columns_to_return = ['input_ids', 'label', 'attention_mask', 'token_type_ids']
    train_dataset.set_format(type='torch', columns=columns_to_return)
    validation_dataset.set_format(type='torch', columns=columns_to_return)


    print(train_dataset)
    print(validation_dataset)
    train_dataloader = DataLoader(
                    train_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    # drop_last=dataloader_drop_last,
                    num_workers=dataloader_num_workers,
                    pin_memory=dataloader_pin_memory,
    )
    if only_train:
        return train_dataloader
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
    )

    return train_dataloader,validation_dataloader


def get_dataloader1(task: str, model_checkpoint: str, dataloader_drop_last: bool = True, shuffle: bool = True,
                   batch_size: int = 16, dataloader_num_workers: int = 2, dataloader_pin_memory: bool = True,
                   tokenizer=None, only_train=False):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]
    train_dataloader = train_dataloader1 = train_dataloader0 = None
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True)

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    validation_name = 'validation'

    if task == "mnli":
        validation_name = "validation_matched"
    if task == "mnli-mm":
        validation_name = "validation_mismatched"

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    columns_to_return = ['input_ids', 'label', 'attention_mask', 'token_type_ids']
    train_dataset = dataset['train']
    if task in ["sst2","mrpc"]:
        train_dataset1 = train_dataset.filter(lambda x: x['label'] == 1)  # 标签为1
        train_dataset0 = train_dataset.filter(lambda x: x['label'] == 0)  # 标签为0
        train_dataset1 = train_dataset1.map(preprocess_function, batched=True)
        train_dataset1.set_format(type='torch', columns=columns_to_return)
        print(train_dataset1)
        train_dataloader1 = DataLoader(
            train_dataset1,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
        train_dataset0 = train_dataset0.map(preprocess_function, batched=True)
        train_dataset0.set_format(type='torch', columns=columns_to_return)
        print(train_dataset0)
        train_dataloader0 = DataLoader(
            train_dataset0,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
    else:
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        train_dataset.set_format(type='torch', columns=columns_to_return)
        print(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=data_collator,
            # drop_last=dataloader_drop_last,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
    if task in ["sst2","mrpc"]:
        return train_dataloader1, train_dataloader0
    else:
        return train_dataloader


def get_dataloader2(task: str, model_checkpoint: str, dataloader_drop_last: bool = True, shuffle: bool = True,
                   batch_size: int = 16, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True,
                   tokenizer=None, only_train=False):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True)

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    validation_name = 'validation'

    if task == "mnli":
        validation_name = "validation_matched"
    if task == "mnli-mm":
        validation_name = "validation_mismatched"
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    train_dataset = dataset['train']

    os.environ['data_num'] = str(len(train_dataset))

    validation_dataset = dataset[validation_name]
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    validation_dataset = validation_dataset.map(preprocess_function, batched=True)

    columns_to_return = ['input_ids', 'label', 'attention_mask', 'token_type_ids']
    train_dataset.set_format(type='torch', columns=columns_to_return)
    validation_dataset.set_format(type='torch', columns=columns_to_return)
    from datasets import Dataset
    train_dataset = Dataset.from_dict({'input_ids': train_dataset['input_ids'],
                                       'label': train_dataset['label'],
                                       'attention_mask': train_dataset['attention_mask'],
                                       'token_type_ids': train_dataset['token_type_ids'],
                                       'idx': list(range(len(train_dataset)))})
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
    )

    if only_train:
        return train_dataloader
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
    )
    return train_dataloader, validation_dataloader, train_dataset

def get_dataloader3(task: str, model_checkpoint: str, dataloader_drop_last: bool = True, shuffle: bool = True,
                   batch_size: int = 16, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True,
                   tokenizer=None, only_train=False):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True)

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    validation_name = 'validation'

    if task == "mnli":
        validation_name = "validation_matched"
    if task == "mnli-mm":
        validation_name = "validation_mismatched"

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    train_dataset = dataset['train']

    os.environ['data_num'] = str(len(train_dataset))

    validation_dataset = dataset[validation_name]
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    validation_dataset = validation_dataset.map(preprocess_function, batched=True)

    columns_to_return = ['input_ids', 'label', 'attention_mask', 'token_type_ids']
    train_dataset.set_format(type='torch', columns=columns_to_return)
    validation_dataset.set_format(type='torch', columns=columns_to_return)
    from datasets import Dataset
    idx = list(range(len(train_dataset)))
    train_dataset = Dataset.from_dict({'input_ids': train_dataset['input_ids'],
                                       'label': train_dataset['label'],
                                       'attention_mask': train_dataset['attention_mask'],
                                       'token_type_ids': train_dataset['token_type_ids'],
                                       'idx': idx})
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
    )
    train_dataloader1 = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
    )
    if only_train:
        return train_dataloader1
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
    )

    return train_dataloader, train_dataloader1, validation_dataloader