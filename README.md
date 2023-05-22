# level2_klue-nlp-06
level2_klue-nlp-06 created by GitHub Classroom


# 📽️프로젝트 개요

## 🧶 Relation Extraction

- 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제이다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요하다.
- 대회의 목표는 문장 내 두 단어(entity)의 관계를 파악하여 30가지의 Label로 잘 분류하는 모델을 학습시키는 것이다.

## 📇 Data

- train.csv : 총 32470개
- test_data.csv : 총 7765개
- Label : 총 30개의 class

## 📑 Metric

- KLUE-RE evaluation metric을 그대로 사용하며, 둘 중 micro F1 score가 우선시됨.
    1. no_relation class를 제외한 micro F1 score
        
        
        $\mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$
        
        $\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$
        
        $\mathrm{F1 \ score} = 2 \times\frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$
        
        
    2. 모든 class에 대한 area under the precision-recall curve(AUPRC)

# 👨‍👩‍👧‍👦 프로젝트 팀 구성 및 역할

- **김민호** : 모델 구조 및 손실 함수 분석
- **김성은** : main 실행 코드 작성, 데이터 전처리, 모델 커스텀
- **김지현** : 전처리 방법 제시, 모델 구조 분석, 커스텀 모델 구현, base setting 기여 및 앙상블
- **서가은** : 하이퍼 파라미터 튜닝 및 다양한 모델 실험
- **홍영훈** : 전처리 방법 제시 및 모델 예측 결과 분석

# 🗂️ 파일 구조

```python
├── src
│   ├── dict_label_to_num.pkl
│   ├── dict_num_to_label.pkl
│   ├── train.py
│   ├── inference.py
│		└── ensemble.py
│		└── hp_train.py
│
│   
├── utils
│   ├── preprocessing.py : tokenizing 이전까지의 전처리 함수를 저장하는 함수
│   ├── tokenizing.py : dataset 이전까지 담당하는 함수들 모아두는 곳
│   ├── metric.py : 메트릭 관련 함수들 모아두는 곳
│   └── load_data.py : 전처리와 데이터셋 구성을 위한 함수 코드!
│
├── result
│   └── {run_name} : 모델 결과
│       └── best_model : 모델 저장하는 곳
│
├── data
│   ├── test
│   │   └── test_data.csv
│   ├── train
│   │   ├── train_original.csv
│   │   ├── train.csv
│   │   └── dev.csv
│   └── prediction
│       └── sample_submission.csv
│
├── main.py
├── requirements.txt
├── README.md
└── config.yaml
```

# 👀 Wrap-up Report

[https://eojjeol-stones.notion.site/REPORT-09253205d8864f7c8837cee868566702](https://www.notion.so/09253205d8864f7c8837cee868566702)

# ✏️ Usage

### install requirements

```
pip install -r requirements.txt
```

### main.py

```
python main.py # train, inference 모두 실행
python main.py -r train # train 실행
python main.py -r inference # inference 실행 
```
