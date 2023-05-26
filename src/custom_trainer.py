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

