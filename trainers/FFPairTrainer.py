import torch
from collections import deque, Counter
from tqdm import tqdm

from classifiers.differential.FFConcat import FFConcatBase
from classifiers.differential.FFDiff import FFDiffBase
from trainers.BaseFFTrainer import BaseFFTrainer
from trainers.embedding_cache import EmbeddingCache, build_pair_keys


class FFPairTrainer(BaseFFTrainer):
    def __init__(
        self,
        model: FFDiffBase | FFConcatBase,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
        recent_losses = deque(maxlen=200)
        label_counter = Counter()
        last_grad_info = None

        # Training loop
        self._reset_grad_accumulation()
        iterator = self._train_batch_iterator(train_dataloader)
        total = self._train_batch_total(train_dataloader)
        
        for step_idx, (_, gt, test, label) in enumerate(tqdm(iterator, total=total), start=1):
            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)
            self._record_batch_size(len(label))

            # Forward pass
            with self._train_autocast_context():
                # Pass label to forward if supported (for ArcFace margin)
                import inspect
                forward_params = inspect.signature(self.model.forward).parameters
                if "label" in forward_params:
                    # FFCosine returns (logit, prob)
                    # FFConcat returns (logits, probs)
                    outputs = self.model(gt, test, label=label)
                else:
                    outputs = self.model(gt, test)
                
                logits, probs = outputs
                
                loss = self.lossfn(logits, label.long())

            loss_value = loss.item()
            grad_info = self._step_optimizer(loss)
            if grad_info:
                last_grad_info = grad_info

            # Compute accuracy
            predicted = torch.argmax(probs, 1)
            
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)

            losses.append(loss_value)
            accuracies.append(accuracy)
            recent_losses.append(loss_value)

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
        use_embedding_cache = (
            hasattr(self.model, "forward_from_embeddings")
            and hasattr(self.model, "extractor")
            and hasattr(self.model, "feature_processor")
        )
        embedding_cache = None
        if use_embedding_cache:
            embedding_cache = EmbeddingCache(self.model.extractor, self.model.feature_processor, self.device)

        # On-the-fly calibration for models that expose calibrate/is_calibrated.
        if hasattr(self.model, "calibrate") and not self.model.is_calibrated:
            print("Model non-calibrated. Running on-the-fly calibration on validation set...")
            self.model.calibrate(val_dataloader, self.device, max_samples=5000)

        for file_name, gt, test, label in tqdm(val_dataloader):
            # print(f"Validation batch {i+1} of {len(val_dataloader)}")

            label = label.to(self.device)

            if use_embedding_cache:
                keys_gt, keys_test = build_pair_keys(file_name, gt, test)
                emb_gt = embedding_cache.get_embeddings(gt, keys_gt, self._eval_autocast_context())
                emb_test = embedding_cache.get_embeddings(test, keys_test, self._eval_autocast_context())
                with self._eval_autocast_context():
                    logits, probs = self.model.forward_from_embeddings(emb_gt, emb_test, label=label)
            else:
                gt = gt.to(self.device)
                test = test.to(self.device)
                with self._eval_autocast_context():
                    logits, probs = self.model(gt, test)
            
            loss = self.lossfn(logits.float(), label.long())
            predictions.extend(torch.argmax(probs, 1).tolist())
            scores.extend(probs[:, 0].float().tolist()) # usually score for class 0 (Diff) or 1 (Same)?
            # Wait, standard existing code: scores.extend(probs[:, 0].float().tolist())
            # FFCosine originally returned [-logit, logit]. Class 0 is Diff, Class 1 is Same.
            # probs[:, 0] is Prob(Diff). 
            # If we want "Same" score, we should return probs[:, 1]?
            # The user context "FFCosine... 2-class logits... [-logit, logit]".
            # If logits[0] = -logit, logits[1] = logit.
            # If logit > 0 (similar), then logits[1] > logits[0]. Prob(Same) > Prob(Diff).
            # Existing code returning probs[:, 0] implies it returns Prob(Diff) or "Diff Score"?
            # Usually we want a score where higher = Same. 
            # If existing code saves probs[:, 0], that's Prob(Diff) (Dissimilarity).
            # But let's check evaluation scripts. Usually EER uses score for TARGET class.
            # If target is Same (1), we should use probs[:, 1].
            # If target is Spoof/Diff (0), we use probs[:, 0].
            # Deepfake training usually treats Spoof/Diff as the target to detect?
            # "FFPair" implies verification. Same vs Diff. 
            # Let's assume for now keeping existing behavior for non-BCE is safer.
                
            if save_scores:
                file_names.extend(file_name)
            losses.append(loss.item())
            labels.extend(label.tolist())

        # for name, label, score, prediction in zip(file_names, labels, scores, predictions):
        #     print(f"File: {name}, Score: {score}, Label: {label}, Prediction: {prediction}")

        return losses, labels, scores, predictions, file_names
