from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import *


class denseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate=0.1):
        super(denseLayer, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        x = self.tanh(x)
        return self.linear(x)


class Custom_ModelForSequenceClassification(BertPreTrainedModel):
    # classification Head with entity tokens
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.config.output_attentions = True
        # config._name_or_path
        self.bert = BertModel(self.config)
        self.tokenizer = None

        # AutoModel
        self.cls_fc = denseLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc = denseLayer(
            config.hidden_size, config.hidden_size, 0.1)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)

        # Initialize weights and apply final processing

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        batch_size = len(input_ids)

        sbj_entity_hidden_states = []
        obj_entity_hidden_states = []
        s_marker = self.tokenizer.convert_tokens_to_ids('@')
        o_marker = self.tokenizer.convert_tokens_to_ids('#')

        for i in range(batch_size):
            L = torch.where(input_ids[i] == s_marker)[0].tolist()
            L2 = torch.where(input_ids[i] == o_marker)[0].tolist()

            sbj_entity_hidden_states.append(torch.mean(
                outputs[0][i][[i for i in range(L[0], L[1]+1)]], dim=0))
            obj_entity_hidden_states.append(torch.mean(
                outputs[0][i][[i for i in range(L2[0], L2[1]+1)]], dim=0))

        sbj_entity_hidden_states = torch.stack(sbj_entity_hidden_states, dim=0)
        obj_entity_hidden_states = torch.stack(obj_entity_hidden_states, dim=0)

        # dropout -> tanh -> denselayer
        sbj_entity_hidden_states = self.entity_fc(sbj_entity_hidden_states)
        obj_entity_hidden_states = self.entity_fc(obj_entity_hidden_states)

        # concat -> label
        # pooled_output = self.cls_fc(outputs[1])
        pooled_output = outputs.pooler_output
        concat_hidden_states = torch.cat(
            [pooled_output, sbj_entity_hidden_states, obj_entity_hidden_states], dim=-1)
        pooled_output = self.dropout(concat_hidden_states)
        logits = self.classifier(pooled_output)

        # outputs = (logits,) + outputs[2:]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Custom_sep_BertForSequenceClassification(BertPreTrainedModel):
    # classification Head with [sep] tokens
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.pooler = BertPooler(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        '''
        token position 찾아서 해당 position 의 hidden state 를 concat
        '''

        batch_size = len(input_ids)

        sep_token_positions = []
        for i in range(batch_size):
            sep_token_positions.append(
                [0]+torch.where(input_ids[i] == self.tokenizer.sep_token_id)[0].tolist())

        sep_hidden_states = torch.stack(
            [outputs[0][i][sep_token_positions[i]] for i in range(batch_size)], dim=0)
        pooled_output = self.avg_pool(
            sep_hidden_states.transpose(1, 2)).squeeze(-1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Bert_Joint_model(BertPreTrainedModel):
    # multi task model with NER and RE
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.s_labels = config.s_labels
        self.e_labels = config.e_labels
        self.config = config

        # config._name_or_path
        self.bert = BertModel(self.config)
        self.tokenizer = None

        # AutoModel
        self.cls_fc = denseLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc = denseLayer(
            config.hidden_size, config.hidden_size, 0.1)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        # dropout rate default = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.RE_classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.SUB_classifier = nn.Linear(config.hidden_size, config.s_labels)
        self.OBJ_classifier = nn.Linear(config.hidden_size, config.e_labels)

        # Initialize weights and apply final processing

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        batch_size = len(input_ids)
        sbj_entity_hidden_states = []
        obj_entity_hidden_states = []
        s_marker = self.tokenizer.convert_tokens_to_ids('@')
        o_marker = self.tokenizer.convert_tokens_to_ids('#')

        for i in range(batch_size):
            L = torch.where(input_ids[i] == s_marker)[0].tolist()
            L2 = torch.where(input_ids[i] == o_marker)[0].tolist()

            sbj_entity_hidden_states.append(torch.mean(
                outputs[0][i][[i for i in range(L[0], L[1]+1)]], dim=0))
            obj_entity_hidden_states.append(torch.mean(
                outputs[0][i][[i for i in range(L2[0], L2[1]+1)]], dim=0))

        sbj_entity_hidden_states = torch.stack(sbj_entity_hidden_states, dim=0)
        obj_entity_hidden_states = torch.stack(obj_entity_hidden_states, dim=0)

        # dropout -> tanh -> denselayer
        sbj_entity_hidden_states = self.entity_fc(sbj_entity_hidden_states)
        obj_entity_hidden_states = self.entity_fc(obj_entity_hidden_states)

        # concat -> label
        pooled_output = self.cls_fc(outputs.pooler_output)
        concat_hidden_states = torch.cat(
            [pooled_output, sbj_entity_hidden_states, obj_entity_hidden_states], dim=-1)

        RE_logits = self.RE_classifier(concat_hidden_states)
        S_logits = self.SUB_classifier(sbj_entity_hidden_states)
        O_logits = self.OBJ_classifier(obj_entity_hidden_states)

        loss = None
        RE_loss = None
        S_loss = None
        O_loss = None

        if labels is not None:
            weight = nn.Parameter(torch.Tensor([0.6, 0.15, 0.25]))

            self.config.problem_type = "single_label_classification"
            if self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                RE_loss = loss_fct(
                    RE_logits.view(-1, self.num_labels), labels[:, 0].view(-1))
                S_loss = loss_fct(
                    S_logits.view(-1, self.s_labels), labels[:, 1].view(-1))
                O_loss = loss_fct(
                    O_logits.view(-1, self.e_labels), labels[:, 2].view(-1))
            loss = weight[0]*RE_loss + weight[1]*S_loss + weight[2]*O_loss

            labels = labels[:, 0].view(-1)

        if not return_dict:
            output = (RE_logits,) + outputs[2:]
            return ((RE_loss,) + output) if RE_loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=RE_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
