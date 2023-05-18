from transformers.models.roberta.modeling_roberta import *
from src.CustomLoss import Loss

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"


class CustomRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_sub_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class CustomRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # 기존의 RE task를 위한 logits를 다루기 위한 classifier
        self.classifier = RobertaClassificationHead(config)
        
        # no_relation을 구분하는 이진 분류를 위한 classifier
        self.num_sub_labels = config.num_sub_labels
        self.binary_classifier = CustomRobertaClassificationHead(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        
        # 기존의 RE 태스크를 위한 logits
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        # no_relation을 구분하는 이진 분류를 위한 logits
        binary_sequence_output = outputs[0]
        binary_logits = self.binary_classifier(binary_sequence_output)

        # Loss function 결정 및 loss 계산
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
            
            # 우리의 Task에 적용되는 Loss
            elif self.config.problem_type == "single_label_classification":
                loss_fct = Loss(
                    loss_type="focal_loss",
                    samples_per_class=self.config.samples_per_class,
                    class_balanced=True
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # binary classification을 위한 label 정의와 loss 계산
        total_loss = None
        binary_loss = None
        if labels is not None:
            binary_labels = [0 for _ in range(len(labels))]
            for idx, label in enumerate(labels):
                if label != 0:
                    binary_labels[idx] = 1
            binary_labels = torch.Tensor(binary_labels).to(device=self.device).long()
            
            binary_loss_fct = Loss(
                loss_type="focal_loss",
                samples_per_class=self.config.sub_samples_per_class,
                class_balanced=True
            )
            binary_loss = binary_loss_fct(binary_logits.view(-1, self.num_sub_labels), binary_labels.view(-1))
        
            total_loss = (1-self.config.sub_task_weight) * loss + self.config.sub_task_weight * binary_loss

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
