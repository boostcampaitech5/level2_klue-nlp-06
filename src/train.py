import os
from utils import *
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, set_seed

import torch
import pickle as pickle
# For wandb setting
'''
terminal 에서,
pip install wandb
wandb login
'''
import wandb
import datetime
import shutil
import yaml
from src.bert_custom_clf import Custom_ModelForSequenceClassification, Bert_Joint_model
from src.roberta_custom_clf import Custom_RobertaForSequenceClassification, Roberta_Joint_model

set_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def label_to_num(label):
    num_label = []
    with open('./src/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

def s_label_to_num(label):
    num_label = []
    with open('./src/dict_slabel_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

def o_label_to_num(label):
    num_label = []
    with open('./src/dict_olabel_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

def train(CFG, save_path):
    # load model and tokenizer
    MODEL_NAME = CFG.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    print('-- preprocessing')
    print('\n'.join(['* '+str(p) for p in CFG.data.preprocessing]))
    print('-- preprocessing dataset')
    print('* '+str(CFG.data.pre_dataset),'\n')
    train_dataset = load_data("./data/train/train_original.csv", CFG.data.pre_dataset, CFG.data.preprocessing)
    dev_dataset = load_data("./data/train/dev.csv", CFG.data.pre_dataset, CFG.data.preprocessing)

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # For NER
    train_slabel = s_label_to_num(train_dataset['subject_entity_type'].values)
    dev_slabel = s_label_to_num(dev_dataset['subject_entity_type'].values)
    
    train_olabel = o_label_to_num(train_dataset['object_entity_type'].values)
    dev_olabel = o_label_to_num(dev_dataset['object_entity_type'].values)
    
    # tokenizing dataset
    tokenized_train = globals()[CFG.data.tokenizer.tokenizing](train_dataset, tokenizer, CFG.data.tokenizer.max_len)
    tokenized_dev = globals()[CFG.data.tokenizer.tokenizing](dev_dataset, tokenizer, CFG.data.tokenizer.max_len)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label, train_slabel, train_olabel)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label, dev_slabel, dev_olabel)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30
    model_config.s_labels = 2
    model_config.e_labels = 6
    
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    
    # custom model
    if CFG.model.mode == 'joint':
        if MODEL_NAME.find('roberta')== -1:
            print("model: Bert joint model")
            model = Bert_Joint_model.from_pretrained(MODEL_NAME, config = model_config)
        else:
            print("model: roBerta joint model")
            model = Roberta_Joint_model.from_pretrained(MODEL_NAME, config = model_config)
    else: 
        if MODEL_NAME.find('roberta')== -1:
            print("model: Bert Custom model")
            model = Custom_ModelForSequenceClassification.from_pretrained(MODEL_NAME, config = model_config)
        else:
            print("model: roBerta Custom model")
            model = Custom_RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config = model_config)
    model.tokenizer = tokenizer

    print(model.config)
    model.parameters
    
    # model.model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=save_path,         # output directory
        # output_dir = utils.make_run_name
        save_total_limit=1,             # number of total save model.
        save_steps=CFG.train.save_steps,                # model saving step.
        num_train_epochs=CFG.train.epochs,            # total number of training epochs
        learning_rate=CFG.train.LR,             # learning_rate
        per_device_train_batch_size=CFG.train.batch_size, # batch size per device during training
        per_device_eval_batch_size=CFG.train.batch_size,  # batch size for evaluation
        warmup_steps=500,               # number of warmup steps for learning rate scheduler
        weight_decay=0,              # strength of weight decay 0.1 -> 0
        logging_dir=f'{save_path}/logs',   # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps',    # evaluation strategy to adopt during training
        warmup_ratio=0, # 0.6 -> 0 
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=CFG.train.eval_steps,         # evaluation step.
        load_best_model_at_end=True,
        metric_for_best_model='micro f1 score',
        lr_scheduler_type='cosine',

        # wandb loggging 추가
        report_to="wandb",  # enable logging to W&B
    )

    # For wandb
    wandb.init(project=MODEL_NAME.replace(r'/', '_'), name=save_path[10:])
    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                 # training arguments, defined above
        train_dataset=RE_train_dataset,     # training dataset
        eval_dataset=RE_dev_dataset,        # evaluation dataset
        compute_metrics=compute_metrics     # define metrics function
    )
    # train model
    trainer.train()
    model.save_pretrained(f'{save_path}/best_model')

    wandb.finish()


def main(CFG, save_path):
    train(CFG, save_path)
    shutil.rmtree('./wandb')

if __name__ == '__main__':
    main()
