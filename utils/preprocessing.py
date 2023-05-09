import pandas as pd


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = i[1:-1].split(',')[0].split(':')[1]
        j = j[1:-1].split(',')[0].split(':')[1]

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



def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset_with_wordtype(pd_dataset)

    return dataset
