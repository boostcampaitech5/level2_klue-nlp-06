from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.roberta.modeling_roberta import *


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
    
class CRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class Custom_RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config            

        #config._name_or_path
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = CRobertaClassificationHead(config.hidden_size*3, config.num_labels, config)
        self.tokenizer = None

        # layer
        self.cls_fc = CRobertaClassificationHead(config.hidden_size, config.hidden_size, config)
        self.entity_fc = CRobertaClassificationHead(config.hidden_size, config.hidden_size, config)
        
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

        outputs = self.roberta(
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
            sbj_entity_hidden_states.append(torch.mean(outputs.last_hidden_state[i][[j for j in range(L[0], L[1]+1)]], dim=0))
            obj_entity_hidden_states.append(torch.mean(outputs.last_hidden_state[i][[j for j in range(L2[0], L2[1]+1)]], dim=0))

        sbj_entity_hidden_states = torch.stack(sbj_entity_hidden_states, dim=0)
        obj_entity_hidden_states = torch.stack(obj_entity_hidden_states, dim=0)

        # dropout -> tanh -> denselayer
        sbj_entity_hidden_states = self.entity_fc(sbj_entity_hidden_states)
        obj_entity_hidden_states = self.entity_fc(obj_entity_hidden_states)
        
        # concat -> label
        pooled_output = self.cls_fc(outputs.last_hidden_state[:,0,:])
        concat_hidden_states = torch.cat([pooled_output, sbj_entity_hidden_states, obj_entity_hidden_states], dim=-1)
        logits = self.classifier(concat_hidden_states)

        #outputs = (logits,) + outputs[2:] 
        
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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


class Roberta_Joint_model(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.s_labels = config.s_labels
        self.e_labels = config.e_labels
        self.config = config
        
        #config._name_or_path
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.RE_classifier = CRobertaClassificationHead(config.hidden_size*3, config.num_labels, config)
        self.SUB_classifier = CRobertaClassificationHead(config.hidden_size, config.s_labels, config)
        self.OBJ_classifier = CRobertaClassificationHead(config.hidden_size, config.e_labels, config)
        self.tokenizer = None

        # layer
        self.cls_fc = CRobertaClassificationHead(config.hidden_size, config.hidden_size, config)
        self.entity_fc = CRobertaClassificationHead(config.hidden_size, config.hidden_size, config)
        
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
        
        outputs = self.roberta(
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
            sbj_entity_hidden_states.append(torch.mean(outputs.last_hidden_state[i][[j for j in range(L[0], L[1]+1)]], dim=0))
            obj_entity_hidden_states.append(torch.mean(outputs.last_hidden_state[i][[j for j in range(L2[0], L2[1]+1)]], dim=0))

        sbj_entity_hidden_states = torch.stack(sbj_entity_hidden_states, dim=0)
        obj_entity_hidden_states = torch.stack(obj_entity_hidden_states, dim=0)

        # dropout -> tanh -> denselayer
        sbj_entity_hidden_states = self.entity_fc(sbj_entity_hidden_states)
        obj_entity_hidden_states = self.entity_fc(obj_entity_hidden_states)
        
        # concat -> label
        pooled_output = self.cls_fc(outputs.last_hidden_state[:,0,:])
        concat_hidden_states = torch.cat([pooled_output, sbj_entity_hidden_states, obj_entity_hidden_states], dim=-1)

        
        RE_logits = self.RE_classifier(concat_hidden_states)
        S_logits = self.SUB_classifier(sbj_entity_hidden_states)
        O_logits = self.OBJ_classifier(obj_entity_hidden_states)
                
        RE_loss = None
        S_loss = None
        O_loss = None

        if labels is not None:
            self.config.problem_type = "single_label_classification"
            if self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                RE_loss = loss_fct(RE_logits.view(-1, self.num_labels), labels[:,0].view(-1))
                S_loss = loss_fct(S_logits.view(-1, self.s_labels), labels[:,1].view(-1))
                O_loss = loss_fct(O_logits.view(-1, self.e_labels), labels[:,2].view(-1))

        
        if not return_dict:
            output = (RE_logits,) + outputs[2:]
            return ((RE_loss,) + output) if RE_loss is not None else output
        labels = labels[:,0].view(-1)
        
        weight = nn.Parameter(torch.Tensor([0.6, 0.15, 0.25]))
        return SequenceClassifierOutput(
            loss= weight[0]*RE_loss + weight[1]*S_loss + weight[2]*O_loss,
            logits=RE_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


