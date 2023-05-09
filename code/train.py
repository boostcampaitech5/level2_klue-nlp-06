import sys_setting
import os
from metric import *
from preprocessing import *
from tokenizing import *
from load_data import *
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


set_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tz = datetime.timezone(datetime.timedelta(hours=9))
day_time = datetime.datetime.now(tz=tz)
run_name = day_time.strftime('%m%d%H%M%S')

dir_path = f'./results/{run_name}'
dir_path_log = f'./results/{run_name}/log'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(dir_path_log):
    os.makedirs(dir_path_log)


def label_to_num(label):
    num_label = []
    with open('code/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("./data/train/train.csv")
    dev_dataset = load_data("./data/train/dev.csv")

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset_with_wordtype(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset_with_wordtype(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=f'./results/{run_name}',         # output directory
        save_total_limit=5,             # number of total save model.
        save_steps=1500,                # model saving step.
        num_train_epochs=5,            # total number of training epochs
        learning_rate=5e-5,             # learning_rate
        per_device_train_batch_size=32, # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,               # number of warmup steps for learning rate scheduler
        weight_decay=0.01,              # strength of wfeight decay
        logging_dir=f'./results/{run_name}/logs',   # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps',    # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,         # evaluation step.
        load_best_model_at_end=True,

        # wandb loggging 추가
        report_to="wandb",  # enable logging to W&B
    )

    # For wandb
    wandb.init(project=MODEL_NAME.replace(r'/', '_'), name=run_name)
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
    model.save_pretrained(f'./results/{run_name}/best_model')

    wandb.finish()


def main():
    train()
    shutil.rmtree('./wandb')


if __name__ == '__main__':
    main()
