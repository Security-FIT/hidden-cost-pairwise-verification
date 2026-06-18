import torch
from torch.nn import CosineEmbeddingLoss
from tqdm import tqdm

from classifiers.differential.FFCosine import FFCosineRaw, FFCosineRaw2
from trainers.BaseFFTrainer import BaseFFTrainer
from trainers.embedding_cache import EmbeddingCache, build_pair_keys


class FFCosineRawTrainer(BaseFFTrainer):
    """
    Trainer for `FFCosineRaw` optimized with `torch.nn.CosineEmbeddingLoss`.

    Notes:
      - Pair labels are assumed to be {0, 1} where 1 denotes "same/positive".
      - CosineEmbeddingLoss expects targets in {+1, -1}, so labels are mapped as: 1 -> +1, 0 -> -1.
      - For EER/DET computation elsewhere in the codebase, label=0 is treated as the positive class
        (`pos_label=0`). Therefore we report scores as a dissimilarity value: score = 1 - cosine_similarity,
        where higher score means "more different".
    """

    def __init__(self, model: FFCosineRaw | FFCosineRaw2, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, device)
        # A margin > 0.0 enforces a "gap" for negative pairs.
        margin = float(getattr(model, "cosine_margin", 0.5))
        self.lossfn = CosineEmbeddingLoss(margin=margin)

    @staticmethod
    def _to_cosine_targets(label: torch.Tensor) -> torch.Tensor:
        """
        Convert {0,1} labels to {-1,+1} targets for CosineEmbeddingLoss.
        """
        label_f = label.float()
        return torch.where(label_f > 0.5, torch.ones_like(label_f), -torch.ones_like(label_f))

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        losses: list[float] = []
        accuracies: list[float] = []

        self._reset_grad_accumulation()
        iterator = self._train_batch_iterator(train_dataloader)
        total = self._train_batch_total(train_dataloader)

        for _, gt, test, label in tqdm(iterator, total=total):
            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)
            self._record_batch_size(len(label))

            targets = self._to_cosine_targets(label)

            with self._train_autocast_context():
                similarity, emb_gt, emb_test = self.model(gt, test, label=label, return_embeddings=True)
                loss = self.lossfn(emb_gt.float(), emb_test.float(), targets.float())

            loss_value = loss.item()
            self._step_optimizer(loss)

            predicted = (similarity > 0.0).long()
            correct = (predicted == label.long()).sum().item()
            accuracy = correct / len(label)

            losses.append(loss_value)
            accuracies.append(accuracy)

        self._finalize_optimizer_step()
        return accuracies, losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        losses: list[float] = []
        labels: list[float] = []
        scores: list[float] = []
        predictions: list[int] = []
        file_names: list[str] = []

        use_embedding_cache = (
            hasattr(self.model, "forward_from_embeddings")
            and hasattr(self.model, "extractor")
            and hasattr(self.model, "feature_processor")
        )
        embedding_cache = None
        if use_embedding_cache:
            embedding_cache = EmbeddingCache(self.model.extractor, self.model.feature_processor, self.device)

        for file_name, gt, test, label in tqdm(val_dataloader):
            label = label.to(self.device)
            targets = self._to_cosine_targets(label)

            if use_embedding_cache:
                keys_gt, keys_test = build_pair_keys(file_name, gt, test)
                emb_gt = embedding_cache.get_embeddings(gt, keys_gt, self._eval_autocast_context())
                emb_test = embedding_cache.get_embeddings(test, keys_test, self._eval_autocast_context())
                with self._eval_autocast_context():
                    similarity, proj_gt, proj_test = self.model.forward_from_embeddings(
                        emb_gt, emb_test, label=label, return_embeddings=True
                    )
            else:
                gt = gt.to(self.device)
                test = test.to(self.device)
                with self._eval_autocast_context():
                    similarity, proj_gt, proj_test = self.model(gt, test, label=label, return_embeddings=True)

            loss = self.lossfn(proj_gt.float(), proj_test.float(), targets.float())

            similarity_f = similarity.float()
            predicted = (similarity_f > 0.0).long()
            predictions.extend(predicted.tolist())

            # Higher score => more different => should align with pos_label=0 in BaseFFTrainer.
            scores.extend((1.0 - similarity_f).tolist())

            if save_scores:
                file_names.extend(file_name)

            losses.append(loss.item())
            labels.extend(label.tolist())

        return losses, labels, scores, predictions, file_names
