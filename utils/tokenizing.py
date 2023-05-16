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

def entity_tokens(df, tokenizer, max_len):
    """
    entity 단어 앞, 뒤에 entity start, end token을 subject, object를 구분하여 넣습니다. 
    Ex) 이순신은 조선 중기의 무신이다. -> [SUB] 이순신 [\SUB] 은 [OBJ] 조선 [\OBJ] 중기의 무신이다. 
    """
    sentences = []
    for id, item in df.iterrows():
        if item['subject_begin'] < item['object_begin']:
            sent = item['sentence']
            sent = sent[:item['subject_begin']] + '[SUB]' + sent[item['subject_begin']:]
            sent = sent[:item['subject_end']+6] + '[/SUB]' + sent[item['subject_end']+6:]
            sent = sent[:item['object_begin']+11] + '[OBJ]' + sent[item['object_begin']+11:]
            sent = sent[:item['object_end']+17] + '[/OBJ]' + sent[item['object_end']+17:]
        else :
            sent = item['sentence']
            sent = sent[:item['object_begin']] + '[OBJ]' + sent[item['object_begin']:]
            sent = sent[:item['object_end']+6] + '[/OBJ]' + sent[item['object_end']+6:]
            sent = sent[:item['subject_begin']+11] + '[SUB]' + sent[item['subject_begin']+11:]
            sent = sent[:item['subject_end']+17] + '[/SUB]' + sent[item['subject_end']+17:]
        sentences.append(sent)

    tokens = ['[SUB]', '[/SUB]', '[OBJ]', '[/OBJ]']
    tokenizer.add_tokens(tokens, special_tokens=True)

    tokenized_sentences = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True
    )
    # breakpoint()
    subject_token = '[SUB]'
    object_token = '[OBJ]'
    tokens_index = []
    for i in range(len(tokenized_sentences)):
        tokens = tokenized_sentences[i].tokens
        idx = [0, tokens.index(subject_token), tokens.index(object_token)]
        tokens_index.append(idx)
    # breakpoint()
    return tokenized_sentences, tokens_index
