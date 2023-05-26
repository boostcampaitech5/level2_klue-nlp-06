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
import optuna
from ray import tune

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

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 5e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0., 0.1, 0.2, 0.6]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0., 0.1]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs",[ 3, 4, 5, 10]),
    }

def tune_config_ray(trial):
    return {"learning_rate": tune.grid_search([1e-5, 2e-5, 3e-5, 5e-5]),
            "num_train_epochs": tune.grid_search([3, 4, 5, 6]),
            "per_device_train_batch_size": tune.grid_search([8, 16, 32]),
            "warmup_ratio": tune.grid_search([0., 0.1, 0.2, 0.6]),
            "weight_decay": tune.grid_search([0., 0.1])
            }



def train(CFG, save_path):
    # load model and tokenizer
    MODEL_NAME = CFG.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    print(f'preprocessing mode: {CFG.data.pre_dataset}')
    train_dataset = load_data("./data/train/train_bal.csv", CFG.data.pre_dataset)
    dev_dataset = load_data("./data/train/dev_bal.csv", CFG.data.pre_dataset)

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    print(f'tokenizing mode: {CFG.data.tokenizer.tokenizing}')
    tokenized_train = globals()[CFG.data.tokenizer.tokenizing](train_dataset, tokenizer, CFG.data.tokenizer.max_len)
    tokenized_dev = globals()[CFG.data.tokenizer.tokenizing](dev_dataset, tokenizer, CFG.data.tokenizer.max_len)

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

    def model_init():
        return model

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
        weight_decay=0.01,              # strength of weight decay
        logging_dir=f'{save_path}/logs',   # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps',    # evaluation strategy to adopt during training
        warmup_ratio=0.1,
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
        model_init=model_init,
        args=training_args,                 # training arguments, defined above
        train_dataset=RE_train_dataset,     # training dataset
        eval_dataset=RE_dev_dataset,        # evaluation dataset
        compute_metrics=compute_metrics     # define metrics function
    )

    best_trial = trainer.hyperparameter_search(hp_space=tune_config_ray,
                                               backend='ray',
                                               direction='maximize')

    print(best_trial)

    # train model
    # trainer.train()
    # model.save_pretrained(f'{save_path}/best_model')

    wandb.finish()


def main(CFG, save_path):
    train(CFG, save_path)
    shutil.rmtree('./wandb')

if __name__ == '__main__':
    main()