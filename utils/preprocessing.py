import pandas as pd


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = i[1:-1].split(',')[0].split(':')[1].lstrip(" '").rstrip("'")
        j = j[1:-1].split(',')[0].split(':')[1].lstrip(" '").rstrip("'")

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': dataset['sentence'],
                               'subject_entity': subject_entity, 'object_entity': object_entity, 'label': dataset['label'], })
    return out_dataset

def punc_preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    sentences = []
    for i, j, s in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):        
        sentence = s[:eval(i)['start_idx']]+'@'+s[eval(i)['start_idx']:eval(i)['end_idx']+1]+'@'+s[eval(i)['end_idx']+1:eval(j)['start_idx']]+'#'+s[eval(j)['start_idx']:eval(j)['end_idx']+1]+'#'+s[eval(j)['end_idx']:]

        i = i[1:-1].split(',')[0].split(':')[1].lstrip(" '").rstrip("'")
        j = j[1:-1].split(',')[0].split(':')[1].lstrip(" '").rstrip("'")
        
        subject_entity.append(i)
        object_entity.append(j)
        sentences.append(sentence)
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentences,
                               'subject_entity': subject_entity, 'object_entity': object_entity, 'label': dataset['label'], })
    return out_dataset


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    #dataset = preprocessing_dataset(pd_dataset)
    dataset = punc_preprocessing_dataset(pd_dataset)

    return dataset
