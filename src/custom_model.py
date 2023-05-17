import torch

from transformers import BertModel, BertPreTrainedModel
from torch import nn
from typing import Optional, Union, Tuple
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
    

class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate=0.1):
        super(DenseLayer, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        x = self.tanh(x)
        return self.linear(x)


class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(self.config)
        self.tokenizer = None

        self.subject_fc = DenseLayer(self.config.hidden_size, self.config.hidden_size, 0.1)
        self.object_fc = DenseLayer(self.config.hidden_size, self.config.hidden_size, 0.1)
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

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

        subject_start_token_hidden_states = []
        object_start_token_hidden_states = []

        subject_target_tokens = ['[SPER]', '[SORG]']
        object_target_tokens = ['[OPER]', '[OORG]', '[ODAT]', '[OLOC]', '[OPOH]', '[ONOH]']
        subject_start_token_markers = [self.tokenizer.convert_tokens_to_ids(i) for i in subject_target_tokens]
        object_start_token_markers = [self.tokenizer.convert_tokens_to_ids(i) for i in object_target_tokens]
        
        for i in range(len(input_ids)):
            item = input_ids[i]
            item_list = item.tolist()

            for j in subject_start_token_markers:
                if j in item_list :
                    subject_marker = item_list.index(j)

            for j in object_start_token_markers:
                if j in item_list:
                    object_marker = item_list.index(j)
            # breakpoint()
            subject_start_token_hidden_states.append(outputs[0][i][subject_marker])
            object_start_token_hidden_states.append(outputs[0][i][object_marker])
            
        subject_start_token_hidden_states = torch.stack(subject_start_token_hidden_states, dim=0)
        object_start_token_hidden_states = torch.stack(object_start_token_hidden_states, dim=0)

        subject_start_token_hidden_states = self.subject_fc(subject_start_token_hidden_states)
        object_start_token_hidden_states = self.object_fc(object_start_token_hidden_states)

        pooled_output = outputs[1]
        hidden_states = torch.cat([pooled_output, subject_start_token_hidden_states, object_start_token_hidden_states], dim=-1)
        pooled_output = self.dropout(hidden_states)
        logits = self.classifier(pooled_output)

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

class CustomBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, token_indices):
        pooled_output = torch.cat([hidden_states[:, index, :] for index in token_indices], dim=1)

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output