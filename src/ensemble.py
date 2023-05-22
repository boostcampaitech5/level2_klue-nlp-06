import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import torch.nn.functional as F
import torch
import pickle as pickle
from collections import Counter
from tqdm import tqdm
import argparse

# ML 내부에 inferences 라는 폴더 생성해 주시고, 해당 inferences 폴더 내부에 앙상블 하고자 하는 output.csv 파일을 넣으면
# 자동으로 긁어와서 ensemble_output.csv 파일을 생성하게 됩니다.
# *주의* inferences 폴더 내부의 파일은 <파일이름_score.csv> 형식을 지켜주셔야 합니다.
# ex) inferences/output_0.97.csv
# '_' 를 기준으로 filename 과 score 를 인식하기 때문에 언더바 사용에 주의해주세요
# inferences 폴더와 내부 파일 생성 후에는 그냥 python code/ensemble.py 로 작동하시면 됩니다~!
# -> python src/ensemble.py -m sw
# mode 종류
# sw - softvoting with score weight
# ss - softvoting with softmax
# else - hard voting

class Ensemble():
    def __init__(self):
        self.files = os.listdir('./inferences')
        self.files = [(file,float(file.replace('.csv',"").split('_')[1])) for file in self.files]
        self.num_files = len(self.files) #(filename, score)
        self.scores = torch.Tensor([inference[1] for inference in self.files])
        self.inf_list = [pd.read_csv('./inferences/'+inference[0])['probs'].apply(lambda row:eval(row)) for inference in self.files]
        self.file_names = [file[0] for file in self.files]
        print(self.file_names)

    def num_to_label(self, label):
    #숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.

        origin_label = []
        with open('./src/dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pickle.load(f)
        for v in label:
            origin_label.append(dict_num_to_label[v])

        return origin_label

    def soft_voting(self, mode='score'):
        print('Mode : soft voting with', end='')
        if mode == 'softmax':
            print(' softmax')
            scores = F.softmax(self.scores, dim=-1)
        else:
            print(' score weight')
            s = sum(self.scores)
            scores =  torch.Tensor([score/s for score in self.scores])
        for i in range(self.num_files):
            self.inf_list[i] = self.inf_list[i].apply(lambda lst: [x * scores[i].item() for x in lst])
        
        concatenated_inf = pd.concat(self.inf_list, axis=1)
        concatenated_inf = pd.Series(concatenated_inf.sum(axis = 1))
        for i in tqdm(range(len(concatenated_inf))):
            for j in range(30):
                for k in range(1,self.num_files):
                    concatenated_inf[i][j] += concatenated_inf[i][j+(30*k)]
            concatenated_inf[i] = concatenated_inf[i][:30]
        ensemble_output = concatenated_inf

        output = pd.read_csv('./data/prediction/submission.csv')
        output = output.drop(columns = ['pred_label', 'probs'])
        
        pred_answer =  ensemble_output.apply(lambda row: np.argmax(row, axis=-1))

        output['pred_label'] = self.num_to_label(pred_answer)
        output['probs'] = ensemble_output
        output.to_csv('./ensemble_output.csv', index=False)
        print('finish!')

    def hard_voting(self):
        print('Mode : hard voting')

        concatenated_inf = pd.concat(self.inf_list, axis=1)
        concatenated_inf = pd.Series(concatenated_inf.sum(axis = 1))
        for i in tqdm(range(len(concatenated_inf))):
            for j in range(30):
                for k in range(1,self.num_files):
                    concatenated_inf[i][j] += concatenated_inf[i][j+(30*k)]
            concatenated_inf[i] = concatenated_inf[i][:30]
        ensemble_output = concatenated_inf

        output = pd.read_csv('./data/prediction/submission.csv')
        output = output.drop(columns = ['pred_label', 'probs'])
        
        pred_answer =  [self.inf_list[i].apply(lambda row: np.argmax(row, axis=-1)) for i in range(self.num_files)]
        pred_answer = [Counter([pred_answer[i][j] for i in range(self.num_files)]).most_common(1)[0][0] for j in range(len(self.inf_list[0]))]


        output['pred_label'] = self.num_to_label(pred_answer)
        output['probs'] = ensemble_output
        output.to_csv('./ensemble_output.csv', index=False)
        print('finish!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default="sw")
    args = parser.parse_args()
    mode = args.mode
    e = Ensemble()
    if mode == 'sw':
        e.soft_voting()
    elif mode == 'ss':
        e.soft_voting(mode='softmax')
    else:
        e.hard_voting()
