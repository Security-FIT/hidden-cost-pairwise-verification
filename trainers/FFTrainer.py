import torch
from tqdm import tqdm
import numpy as np

from classifiers.single_input.FF import FF
from trainers.BaseFFTrainer import BaseFFTrainer


class FFTrainer(BaseFFTrainer):
    def __init__(
        self, model: FF, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(model, device)

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        """
        Train the model on the given dataloader for one epoch
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        return: Tuple(lists of accuracies, list of losses)
        """
        # For accuracy computation in the epoch
        losses = []
        accuracies = []

        # Training loop
        self._reset_grad_accumulation()
        iterator = self._train_batch_iterator(train_dataloader)
        total = self._train_batch_total(train_dataloader)
        for _, wf, label in tqdm(iterator, total=total):

            wf = wf.to(self.device)
            label = label.to(self.device)

            # Forward pass
            with self._train_autocast_context():
                logits, probs = self.model(wf)
                loss = self.lossfn(logits, label.long())
            loss_value = loss.item()
            self._step_optimizer(loss)

            # Compute accuracy
            predicted = torch.argmax(probs, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)

            losses.append(loss_value)
            accuracies.append(accuracy)

        self._finalize_optimizer_step()
        return accuracies, losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        losses = []
        labels = []
        scores = []
        predictions = []
        file_names = []

        for file_name, wf, label in tqdm(val_dataloader):
            wf = wf.to(self.device)
            label = label.to(self.device)

            with self._eval_autocast_context():
                logits, probs = self.model(wf)
            if any(torch.isnan(label)):
                loss = np.inf
            else:
                loss = self.lossfn(logits.float(), label.long()).item()

            predictions.extend(torch.argmax(probs, 1).tolist())

            if save_scores:
                file_names.extend(file_name)
            losses.append(loss)
            labels.extend(label.tolist())
            if getattr(self.model, "num_classes", 2) == 2:
                scores.extend(probs[:, 0].float().tolist())
            else:
                scores.extend(probs.max(dim=1).values.float().tolist())

        return losses, labels, scores, predictions, file_names
