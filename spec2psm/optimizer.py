from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


class Optimizer:
    def __init__(self, model, lr, weight_decay, warmup_steps, total_steps):
        """
        Initializes the optimizer and learning rate scheduler.

        Parameters:
        - model: The model whose parameters will be optimized.
        - lr: Learning rate for the optimizer.
        - weight_decay: Weight decay for the optimizer.
        - warmup_steps: Number of warm-up steps for the scheduler.
        - total_steps: Total number of steps for the training.
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Initialize the optimizer
        self.optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Initialize the warm-up scheduler
        self.warmup_scheduler = LambdaLR(
            self.optimizer, lr_lambda=lambda step: step / self.warmup_steps if step < self.warmup_steps else 1
        )

        # Initialize the cosine decay scheduler
        self.cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.total_steps - self.warmup_steps)

        # Combine the schedulers
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[self.warmup_scheduler, self.cosine_scheduler], milestones=[self.warmup_steps]
        )

    @classmethod
    def load_from_state_dict(cls, model, state_dict, lr, weight_decay, warmup_steps, total_steps):
        """
        Loads the optimizer and scheduler states from state dictionaries.

        Parameters:
        - model: The model whose parameters will be optimized.
        - state_dict: A dictionary containing 'optimizer' and 'scheduler' state_dicts.
        - lr: Learning rate for the optimizer (needed to initialize).
        - weight_decay: Weight decay for the optimizer (needed to initialize).
        - warmup_steps: Number of warm-up steps for the scheduler (needed to initialize).
        - total_steps: Total number of steps for the training (needed to initialize).

        Returns:
        - An instance of the Optimizer class with restored states.
        """
        # Initialize the optimizer instance
        instance = cls(model, lr, weight_decay, warmup_steps, total_steps)

        # Load the optimizer state
        instance.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        # Load the scheduler state
        instance.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        return instance

    def state_dict(self):
        """
        Returns a state dictionary containing the states of the optimizer and scheduler.
        """
        return {'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict()}
