from torch import nn
from transformers import Trainer
from transformers.optimization import get_scheduler

import torch


class CustomTrainer(Trainer):

        def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
            """
            Learning rate scheduler를 커스텀할 수 있습니다. 

            Args:
                num_training_steps (int): The number of training steps to do.
            """
            if self.lr_scheduler is None:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            """
            아래와 같이 원하는 Learning rate scheduler를 설정할 수 있습니다. 

            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=500,
                gamma=0.5
            )
            """

            return self.lr_scheduler

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            loss 함수를 커스텀할 수 있습니다. 
            """
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
