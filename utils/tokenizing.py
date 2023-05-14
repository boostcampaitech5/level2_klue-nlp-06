def tokenized_dataset(dataset, tokenizer, max_len):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    return tokenized_sentences


def tokenized_dataset_with_wordtype(df, tokenizer, max_len):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    sents = []
    
    for idx, row in df.iterrows():
        sent = ''
        s_st = row['subject_begin']
        s_end = row['subject_end']
        o_st = row['object_begin']
        o_end = row['object_end']
    
        if s_st < o_st:
            sent = row['sentence']
            sent = sent[:s_st] + '[' +row['subject_entity_type'] + ']' + sent[s_st:]
            sent = sent[:s_end+7] + '[/' +row['subject_entity_type'] + ']' + sent[s_end+7:]
            sent = sent[:o_st+13] + '[' +row['object_entity_type'] + ']' + sent[o_st+13:]
            sent = sent[:o_end+20] + '[/' +row['object_entity_type'] + ']' + sent[o_end+20:]
        else :
            sent = row['sentence']
            sent = sent[:o_st] + '[' +row['object_entity_type'] + ']' + sent[o_st:]
            sent = sent[:o_end+7] + '[/' +row['object_entity_type'] + ']' + sent[o_end+7:]
            sent = sent[:s_st+13] + '[' +row['subject_entity_type'] + ']' + sent[s_st+13:]
            sent = sent[:s_end+20] + '[/' +row['subject_entity_type'] + ']' + sent[s_end+20:]
        
        sents.append(sent)
    
    tokens = ['[SPER]','[/SPER]','[SORG]','[/SORG]',
              '[OPER]','[/OPER]','[OORG]','[/OORG]',
              '[ODAT]','[/ODAT]','[OLOC]','[/OLOC]',
              '[OPOH]','[/OPOH]','[ONOH]','[/ONOH]']
    
    tokenizer.add_tokens(tokens,special_tokens=True)
    
    tokenized_sentences = tokenizer(
        sents,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    
    return tokenized_sentences


def tokenized_dataset_with_entity_marker(df, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    sents = []
    
    for idx, row in df.iterrows():
        sent = ''
        s_st = row['subject_begin']
        s_end = row['subject_end']
        o_st = row['object_begin']
        o_end = row['object_end']
    
        if s_st < o_st:
            sent = row['sentence']
            sent = sent[:s_st] + '[' +'SUB' + ']' + sent[s_st:]
            sent = sent[:s_end+7] + '[/' +'SUB' + ']' + sent[s_end+7:]
            sent = sent[:o_st+13] + '[' +'OBJ' + ']' + sent[o_st+13:]
            sent = sent[:o_end+20] + '[/' +'OBJ' + ']' + sent[o_end+20:]
        else :
            sent = row['sentence']
            sent = sent[:o_st] + '[' +'OBJ' + ']' + sent[o_st:]
            sent = sent[:o_end+7] + '[/' +'OBJ' + ']' + sent[o_end+7:]
            sent = sent[:s_st+13] + '[' +'SUB' + ']' + sent[s_st+13:]
            sent = sent[:s_end+20] + '[/' +'SUB' + ']' + sent[s_end+20:]
        
        sents.append(sent)
    
    tokens = ['[SUB]','[/SUB]','[OBJ]','[/OBJ]']
    
    tokenizer.add_tokens(tokens,special_tokens=True)
    
    tokenized_sentences = tokenizer(
        sents,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    
    return tokenized_sentences