import pandas as pd


def default_preprocessing(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    
    dataset['subject_entity_type'] = dataset['subject_entity'].apply(lambda x: eval(x)['type'])
    dataset['object_entity_type'] = dataset['object_entity'].apply(lambda x: eval(x)['type'])
    dataset['subject_begin'] = dataset['subject_entity'].apply(lambda x: eval(x)['start_idx'])
    dataset['subject_end'] = dataset['subject_entity'].apply(lambda x: eval(x)['end_idx'])
    dataset['object_begin'] = dataset['object_entity'].apply(lambda x: eval(x)['start_idx'])
    dataset['object_end'] = dataset['object_entity'].apply(lambda x: eval(x)['end_idx'])
    dataset['subject_entity'] = dataset['subject_entity'].apply(lambda x: eval(x)['word'])
    dataset['object_entity'] = dataset['object_entity'].apply(lambda x: eval(x)['word'])

    return dataset


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

    sentences = []
    for sb, se, ob, oe, s in zip(dataset['subject_begin'], dataset['subject_end'], dataset['object_begin'], dataset['object_end'], dataset['sentence']):
        # sentence 변환
        if sb < ob:
            sentence = s[:sb]+'@'+s[sb:se+1]+'@'+s[se+1:ob]+'#'+s[ob:oe+1]+'#'+s[oe+1:]
        else:
            sentence = s[:ob]+'#'+s[ob:oe+1]+'#'+s[oe+1:sb]+'@'+s[sb:se+1]+'@'+s[se+1:]
        
        sentences.append(sentence)
    dataset.drop(columns='sentence', inplace=True)
    dataset['sentence'] = sentences
    
    return dataset

def punc_typed_preprocessing_dataset(dataset):
    """ 
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
    typed punctuation entity marker 를 추가하여 DataFrame 으로 변경하는 함수 입니다.

    Ex) @*ORG*옹진군@은 백령도 간척지에 대규모 #^POH^해당화# 단지를 조성하고, 수확한 열매로 지난해부터 음료와 초콜릿 생산을 시작했다.
    """
    sentences = []
    for sb, se, ob, oe, stype, otype, s in zip(dataset['subject_begin'], dataset['subject_end'], dataset['object_begin'], dataset['object_end'], dataset['subject_entity_type'], dataset['object_entity_type'], dataset['sentence']):
        if sb < ob:
            sentence = s[:sb]+'@'+'*'+stype+'*'+s[sb:se+1]+'@'+s[se+1:ob]+'#'+'^'+otype+'^'+s[ob:oe+1]+'#'+s[oe+1:]
        else:
            sentence = s[:ob]+'#'+'^'+otype+'^'+s[ob:oe+1]+'#'+s[oe+1:sb]+'@'+'*'+stype+'*'+s[sb:se+1]+'@'+s[se+1:]

        sentences.append(sentence)
    dataset.drop(columns='sentence', inplace=True)
    dataset['sentence'] = sentences

    return dataset

def load_data(dataset_dir, ppc_mode):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = default_preprocessing(pd_dataset)
    
    # dataset을 사용하여 원하는 preprocessing 함수를 적용할 수 있습니다.
    ### config 파일에 원하는 preprocessing 함수를 나열 해 주세요 ###
    # DA/ DC 적용 코드

    ### config 파일에 원하는 preprocessing_dataset 함수를 입력 해 주세요 ###
    dataset = globals()[ppc_mode](dataset)
    return dataset
