from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from src.CustomBertModel import CustomBertForSequenceClassification
from src.CustomRobertaModel import CustomRobertaForSequenceClassification
from torch.utils.data import DataLoader
from utils import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm


def inference(model, tokenized_sent, device):
    """
      test dataset을 DataLoader로 만들어 준 후,
      batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def num_to_label(label):
    """
      숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('./src/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer, CFG):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, CFG.data.pre_dataset)
    test_label = list(map(int, test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = globals()[CFG.data.tokenizer.tokenizing](test_dataset, tokenizer, CFG.data.tokenizer.max_len)
    return test_dataset['id'], tokenized_test, test_label

def load_dev_dataset(dataset_dir, tokenizer, CFG):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, CFG.data.pre_dataset)
    test_dataset['label'] = 100
    test_label = list(map(int, test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test =  globals()[CFG.data.tokenizer.tokenizing](test_dataset, tokenizer, CFG.data.tokenizer.max_len)
    return test_dataset['id'], tokenized_test, test_label

def save_prediction(model, dev_dir, device, tokenizer, save_path, CFG):
    """
    dev data의 예측값 반환 후 저장하는 함수.
    """

    dev_id, dev_dataset, dev_label = load_dev_dataset(dev_dir, tokenizer, CFG)
    Re_dev_dataset = RE_Dataset(dev_dataset, dev_label)
    pred_dev_answer, output_dev_prob = inference(model, Re_dev_dataset, device)
    pred_dev_answer = num_to_label(pred_dev_answer)

    dev_output = pd.DataFrame(
    {'id': dev_id, 'pred_label': pred_dev_answer, 'probs': output_dev_prob, })


    # dev data 예측한 라벨 csv 파일 형태로 저장
    dev_output.to_csv(f'{save_path}/dev_prediction.csv', index=False)


def main(CFG, run_type, save_path):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = CFG.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # load my model
    model_name = CFG.inference.run_name   # 모델 이름 config 파일에서 불러오기
    if run_type == "both":
        model_dir = f"{save_path}/best_model"
    else: # inference만 실행하는 경우
        if CFG.inference.ckpt == 0: #inference 하고자 하는 run dir 안의 best model 사용
            model_dir = f"./results/{model_name}/best_model"
        else:
            model_dir = f"./results/{model_name}/checkpoint-{CFG.inference.ckpt}"

    # load test datset
    test_dataset_dir = "./data/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(
        test_dataset_dir, tokenizer, CFG)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    print("Inference model path :", model_dir)
    # model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = CustomRobertaForSequenceClassification.from_pretrained(model_dir)
    model.parameters
    model.to(device)
    print(model.config)

    # predict dev and save
    get_dev_prediction = CFG.inference.get_dev_pred # dev prediction 파일 만들지 말지 결정하는 변수. config에서 불러오기.
    if get_dev_prediction==True:
        save_prediction(model, './data/train/dev.csv', device, tokenizer, model_dir, CFG)

    # predict answer
    pred_answer, output_prob = inference(
        model, Re_test_dataset, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장
    if run_type == "both":
        output.to_csv(f'{save_path}/submission.csv', index=False)
    else:
        output.to_csv(f'{model_dir}/submission.csv', index=False)

    #### 필수!! ##############################################
    print('---- Finish! ----')


if __name__ == '__main__':
    main()
