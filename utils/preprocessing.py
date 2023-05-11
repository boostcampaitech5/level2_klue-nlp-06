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


def preprocessing_dataset_with_wordtype(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    dataset['subject_entity_type'] = dataset['subject_entity'].apply(lambda x: 'S' + dict(eval(x))['type'])
    dataset['object_entity_type'] = dataset['object_entity'].apply(lambda x: 'O' + dict(eval(x))['type'])
    dataset['subject_begin'] = dataset['subject_entity'].apply(lambda x: dict(eval(x))['start_idx'])
    dataset['subject_end'] = dataset['subject_entity'].apply(lambda x: dict(eval(x))['end_idx'])
    dataset['object_begin'] = dataset['object_entity'].apply(lambda x: dict(eval(x))['start_idx'])
    dataset['object_end'] = dataset['object_entity'].apply(lambda x: dict(eval(x))['end_idx'])
    
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = i[1:-1].split(',')[0].split(':')[1]
        j = j[1:-1].split(',')[0].split(':')[1]

        subject_entity.append(i)
        object_entity.append(j)
        
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': dataset['sentence'],
                               'subject_entity': subject_entity, 'object_entity': object_entity, 'label': dataset['label'], 
                               'subject_entity_type': dataset['subject_entity_type'],
                               'object_entity_type': dataset['object_entity_type'],
                               'object_end': dataset['object_end'],
                               'object_begin': dataset['object_begin'],
                               'subject_end': dataset['subject_end'],
                               'subject_begin': dataset['subject_begin'],})
    return out_dataset


def punc_preprocessing_dataset(dataset):
    """ 
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
    punctuation entity marker 를 추가하여 DataFrame 으로 변경하는 함수 입니다.

    Ex) @옹진군@은 백령도 간척지에 대규모 #해당화# 단지를 조성하고, 수확한 열매로 지난해부터 음료와 초콜릿 생산을 시작했다.
    """
    subject_entity = []
    object_entity = []
    sentences = []
    for i, j, s in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
        if eval(i)['start_idx'] < eval(j)['start_idx']:
            sentence = s[:eval(i)['start_idx']]+'@'+s[eval(i)['start_idx']:eval(i)['end_idx']+1]+'@'+s[eval(i)['end_idx']+1:eval(j)['start_idx']]+'#'+s[eval(j)['start_idx']:eval(j)['end_idx']+1]+'#'+s[eval(j)['end_idx']+1:]
        else:
            sentence = s[:eval(j)['start_idx']]+'#'+s[eval(j)['start_idx']:eval(j)['end_idx']+1]+'#'+s[eval(j)['end_idx']+1:eval(i)['start_idx']]+'@'+s[eval(i)['start_idx']:eval(i)['end_idx']+1]+'@'+s[eval(i)['end_idx']+1:]


        i = i[1:-1].split(',')[0].split(':')[1].lstrip(" '").rstrip("'")
        j = j[1:-1].split(',')[0].split(':')[1].lstrip(" '").rstrip("'")
        
        subject_entity.append(i)
        object_entity.append(j)
        sentences.append(sentence)
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentences,
                               'subject_entity': subject_entity, 'object_entity': object_entity, 'label': dataset['label'], })
    return out_dataset

def punc_typed_preprocessing_dataset(dataset):
    """ 
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
    punctuation entity marker 를 추가하여 DataFrame 으로 변경하는 함수 입니다.

    Ex) @*ORG*옹진군@은 백령도 간척지에 대규모 #^POH^해당화# 단지를 조성하고, 수확한 열매로 지난해부터 음료와 초콜릿 생산을 시작했다.
    """
    subject_entity = []
    object_entity = []
    sentences = []
    for i, j, s in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
        if eval(i)['start_idx'] < eval(j)['start_idx']:
            sentence = s[:eval(i)['start_idx']]+'@'+'*'+eval(i)['type']+'*'+s[eval(i)['start_idx']:eval(i)['end_idx']+1]+'@'+s[eval(i)['end_idx']+1:eval(j)['start_idx']]+'#'+'^'+eval(j)['type']+'^'+s[eval(j)['start_idx']:eval(j)['end_idx']+1]+'#'+s[eval(j)['end_idx']+1:]
        else:
            sentence = s[:eval(j)['start_idx']]+'#'+'^'+eval(j)['type']+'^'+s[eval(j)['start_idx']:eval(j)['end_idx']+1]+'#'+s[eval(j)['end_idx']+1:eval(i)['start_idx']]+'@'+'*'+eval(i)['type']+'*'+s[eval(i)['start_idx']:eval(i)['end_idx']+1]+'@'+s[eval(i)['end_idx']+1:]


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
    dataset = preprocessing_dataset(pd_dataset) # 해당 부분을 변경하여 원하는 preprocessing 함수를 적용한 dataset 을 얻을 수 있습니다.

    return dataset
