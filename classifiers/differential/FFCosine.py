import torch
import torch.nn as nn
import torch.nn.functional as F

from classifiers.FFBase import FFBase


class FFCosine(nn.Module):
    """
    Feedforward classifier using cosine similarity between the reference (A) and tested (B) embeddings.
    Incorporates an MLP Projection Head ("Cleaning Lens") and Siamese ArcFace Loss.
    """

    def __init__(self, extractor, feature_processor, in_dim: int = 1024):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Unused, kept for interface compatibility with other classifiers.
        """
        super().__init__()
        self.extractor = extractor
        self.feature_processor = feature_processor
        self.in_dim = in_dim

        # Learnable parameters to calibrate the cosine similarity
        # scale (w): initializes to 10 to allow probabilities close to 0 or 1
        # bias (b): initializes to -5 to set a stricter initial threshold than 0
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def forward(self, input_data_ground_truth, input_data_tested):
        # 1. Extract and Process
        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.feature_processor(emb_gt)
        emb_test = self.feature_processor(emb_test)

        # 2. Compute Cosine Similarity (-1 to 1)
        similarity = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)

        # 3. Apply Affine Transformation (Scale & Shift)
        logit = (similarity * self.scale) + self.bias

        # 4. Convert to 2-class logits and probabilities for CrossEntropyLoss
        logits = torch.stack([-logit, logit], dim=1) 
        
        probs = F.softmax(logits, dim=1)

        return logits, probs

    def forward_from_embeddings(self, emb_gt, emb_test, label=None):
        """
        Forward pass using precomputed embeddings (after feature_processor).
        """
        similarity = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)
        logit = (similarity * self.scale) + self.bias
        logits = torch.stack([-logit, logit], dim=1)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def get_param_groups(self):
        """
        Optional optimizer param grouping.

        When SSL extractor fine-tuning is enabled, keep a separate (smaller)
        learning rate for extractor parameters via `self._extractor_lr`.
        """
        if not getattr(self, "_ssl_finetune", False):
            return None

        extractor_lr = getattr(self, "_extractor_lr", None)
        if extractor_lr is None:
            extractor_lr = 1e-6
        extractor_lr = float(extractor_lr)

        extractor_params = [p for p in self.extractor.parameters() if p.requires_grad]
        if not extractor_params:
            return None

        extractor_ids = {id(p) for p in extractor_params}
        other_params = [p for p in self.parameters() if p.requires_grad and id(p) not in extractor_ids]

        return [
            {"params": other_params},
            {"params": extractor_params, "lr": extractor_lr, "group": "extractor"},
        ]


class FFCosineJoint(nn.Module):
    """
    Joint-objective cosine verifier.

    Trains with:
      - Global multiclass CE over generator IDs (shared embedding space anchor)
      - Pairwise CE over same/different labels (verification objective)

    During inference (default forward), behaves like FFCosine and returns only
    pairwise logits/probabilities for drop-in compatibility with existing
    pairwise evaluation code paths.
    """

    def __init__(
        self,
        extractor,
        feature_processor,
        in_dim: int = 1024,
        num_classes: int = 24,
        pair_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.extractor = extractor
        self.feature_processor = feature_processor
        self.in_dim = in_dim
        self.num_classes = int(num_classes)
        self.pair_loss_weight = float(pair_loss_weight)

        # Pairwise cosine calibration (same as FFCosine).
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

        # Global generator-ID head over the shared embedding.
        self.classifier = nn.Linear(in_dim, self.num_classes)
        # Populated by trainer once dataset model names are mapped.
        self.model_label_map: dict[str, int] | None = None

    def _extract_embedding(self, waveforms: torch.Tensor) -> torch.Tensor:
        emb = self.extractor.extract_features(waveforms)
        emb = self.feature_processor(emb)
        return emb

    def _pairwise_head(self, emb_gt: torch.Tensor, emb_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        similarity = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)
        logit = (similarity * self.scale) + self.bias
        logits = torch.stack([-logit, logit], dim=1)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def _global_head(self, emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(emb)

    def forward(self, input_data_ground_truth, input_data_tested, label=None, return_aux: bool = False):
        emb_gt = self._extract_embedding(input_data_ground_truth)
        emb_test = self._extract_embedding(input_data_tested)
        return self.forward_from_embeddings(emb_gt, emb_test, label=label, return_aux=return_aux)

    def forward_from_embeddings(self, emb_gt, emb_test, label=None, return_aux: bool = False):
        logits, probs = self._pairwise_head(emb_gt, emb_test)
        if not return_aux:
            return logits, probs

        cls_logits_gt = self._global_head(emb_gt)
        cls_logits_test = self._global_head(emb_test)
        return logits, probs, cls_logits_gt, cls_logits_test

    def extract_embedding(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self._extract_embedding(waveforms)

    def get_param_groups(self):
        """
        Optional optimizer param grouping.

        When SSL extractor fine-tuning is enabled, keep a separate (smaller)
        learning rate for extractor parameters via `self._extractor_lr`.
        """
        if not getattr(self, "_ssl_finetune", False):
            return None

        extractor_lr = getattr(self, "_extractor_lr", None)
        if extractor_lr is None:
            extractor_lr = 1e-6
        extractor_lr = float(extractor_lr)

        extractor_params = [p for p in self.extractor.parameters() if p.requires_grad]
        if not extractor_params:
            return None

        extractor_ids = {id(p) for p in extractor_params}
        other_params = [p for p in self.parameters() if p.requires_grad and id(p) not in extractor_ids]

        return [
            {"params": other_params},
            {"params": extractor_params, "lr": extractor_lr, "group": "extractor"},
        ]


class FFCosineRaw(nn.Module):
    """
    Siamese head that outputs raw cosine similarity in [-1, 1].

    Intended for optimization with `torch.nn.CosineEmbeddingLoss` (or equivalent margin losses)
    instead of converting the similarity into logits/probabilities.
    """

    def __init__(self, extractor, feature_processor, in_dim: int = 1024, l2_normalize: bool = False):
        """
        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Unused, kept for interface compatibility with other classifiers.
        param l2_normalize: If True, explicitly L2-normalize embeddings after `feature_processor`.
                            This does not change cosine-based losses/scores (they normalize internally),
                            but can stabilize embedding magnitudes for downstream uses (e.g. analysis/visualization).
        """
        super().__init__()
        self.extractor = extractor
        self.feature_processor = feature_processor
        self.in_dim = in_dim
        self.l2_normalize = l2_normalize

    def forward(self, input_data_ground_truth, input_data_tested, label=None, return_embeddings: bool = False):
        # 1. Extract and Process
        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.feature_processor(emb_gt)
        emb_test = self.feature_processor(emb_test)
        if self.l2_normalize:
            emb_gt = F.normalize(emb_gt, p=2, dim=1, eps=1e-12)
            emb_test = F.normalize(emb_test, p=2, dim=1, eps=1e-12)

        # 2. Compute Cosine Similarity (-1 to 1)
        similarity = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)

        if return_embeddings:
            return similarity, emb_gt, emb_test
        return similarity

    def forward_from_embeddings(self, emb_gt, emb_test, label=None, return_embeddings: bool = False):
        """
        Forward pass using precomputed embeddings (after feature_processor).
        """
        if self.l2_normalize:
            emb_gt = F.normalize(emb_gt, p=2, dim=1, eps=1e-12)
            emb_test = F.normalize(emb_test, p=2, dim=1, eps=1e-12)
        similarity = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)
        if return_embeddings:
            return similarity, emb_gt, emb_test
        return similarity


class FFCosineRaw2(FFBase):
    """
    Cosine-similarity head over the FFBase penultimate embedding.

    Uses the same classifier stack as FF/FFCosine1 so it can load FF checkpoints,
    but outputs raw cosine similarity for pairwise training with CosineEmbeddingLoss.
    """

    def __init__(self, extractor, feature_processor, in_dim: int = 1024, num_classes: int = 2, l2_normalize: bool = False):
        super().__init__(extractor, feature_processor, in_dim=in_dim, num_classes=num_classes)
        self.l2_normalize = l2_normalize

    def _penultimate(self, emb: torch.Tensor) -> torch.Tensor:
        return super().extract_embedding(emb)

    def _extract_penultimate(self, waveforms: torch.Tensor) -> torch.Tensor:
        emb = self.extractor.extract_features(waveforms)
        emb = self.feature_processor(emb)
        return self._penultimate(emb)

    def forward(self, input_data_ground_truth, input_data_tested, label=None, return_embeddings: bool = False):
        emb_gt = self._extract_penultimate(input_data_ground_truth)
        emb_test = self._extract_penultimate(input_data_tested)
        if self.l2_normalize:
            emb_gt = F.normalize(emb_gt, p=2, dim=1, eps=1e-12)
            emb_test = F.normalize(emb_test, p=2, dim=1, eps=1e-12)
        similarity = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)
        if return_embeddings:
            return similarity, emb_gt, emb_test
        return similarity

    def forward_from_embeddings(self, emb_gt, emb_test, label=None, return_embeddings: bool = False):
        """
        Forward pass using precomputed embeddings (after feature_processor).
        """
        proj_gt = self._penultimate(emb_gt)
        proj_test = self._penultimate(emb_test)
        if self.l2_normalize:
            proj_gt = F.normalize(proj_gt, p=2, dim=1, eps=1e-12)
            proj_test = F.normalize(proj_test, p=2, dim=1, eps=1e-12)
        similarity = F.cosine_similarity(proj_gt, proj_test, dim=1, eps=1e-8)
        if return_embeddings:
            return similarity, proj_gt, proj_test
        return similarity



class FFMulticlass(nn.Module):
    """
    Standard Feedforward Multiclass Classifier (Pre-trainer).
    Structure: Extractor -> Processor -> Bottleneck (50 dim) -> Linear Head (N classes).
    """
    def __init__(self, extractor, feature_processor, in_dim: int = 1024, 
                 bottleneck_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.extractor = extractor
        self.feature_processor = feature_processor
        self.num_classes = num_classes
        
        # CORRECTED BOTTLENECK:
        # 1. Linear compression (bias=True is fine with LN)
        # 2. LayerNorm (Batch-agnostic, stable for B=16)
        # 3. No Activation
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )

        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, input_data):
        # 1. Extract
        emb = self.extractor.extract_features(input_data)
        emb = self.feature_processor(emb)
        
        # 2. Bottleneck (The 50-dim Fingerprint)
        # We need this variable to be accessible if we were doing feature extraction,
        # but for training the classifier, we just pass it on.
        embedding = self.bottleneck(emb)
        
        # 3. Classify
        logits = self.classifier(embedding)
        
        # Optional: Return probabilities for logging
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        return logits, probs

    def extract_embedding(self, waveforms):
        """
        Return bottleneck embeddings for the given waveforms.
        """
        emb = self.extractor.extract_features(waveforms)
        emb = self.feature_processor(emb)
        return self.bottleneck(emb)


class FFCosine1(FFBase):
    """
    Feedforward cosine verifier using FF penultimate embeddings.
    Structure: Extractor -> Processor -> FF stack (penultimate) -> Cosine -> Scale/Bias.
    Compatible with FF checkpoints (same classifier stack).
    """

    def __init__(self, extractor, feature_processor, in_dim: int = 1024, num_classes: int = 2):
        super().__init__(extractor, feature_processor, in_dim=in_dim, num_classes=num_classes)
        self.in_dim = in_dim

        # Learnable parameters for scaling cosine to logits.
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def _penultimate(self, emb: torch.Tensor) -> torch.Tensor:
        return super().extract_embedding(emb)

    def _extract_penultimate(self, waveforms: torch.Tensor) -> torch.Tensor:
        emb = self.extractor.extract_features(waveforms)
        emb = self.feature_processor(emb)
        return self._penultimate(emb)

    def forward(self, input_data_ground_truth, input_data_tested, label=None):
        # Extract and project through FF stack (penultimate layer).
        emb_gt = self._extract_penultimate(input_data_ground_truth)
        emb_test = self._extract_penultimate(input_data_tested)

        cosine = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)
        scale = torch.clamp(self.scale, max=30.0)
        logit = (cosine * scale) + self.bias

        logits = torch.stack([-logit, logit], dim=1)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def forward_from_embeddings(self, emb_gt, emb_test, label=None):
        """
        Forward pass using precomputed embeddings (after feature_processor).
        """
        proj_gt = self._penultimate(emb_gt)
        proj_test = self._penultimate(emb_test)

        cosine = F.cosine_similarity(proj_gt, proj_test, dim=1, eps=1e-8)
        scale = torch.clamp(self.scale, max=30.0)
        logit = (cosine * scale) + self.bias

        logits = torch.stack([-logit, logit], dim=1)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def extract_embedding(self, waveforms):
        """
        Return penultimate FF embeddings for the given waveforms.
        """
        return self._extract_penultimate(waveforms)


class FFCosine3(nn.Module):
    """
    Antigravity Model (SourceVerifier).
    Structure: Extractor -> Processor -> Bottleneck (50 dim) -> Cosine Similarity -> Scale/Bias.
    Transfers weights from FFMulticlass (inherits extractor, processor, bottleneck).
    """
    def __init__(self, extractor, feature_processor, in_dim: int = 1024, 
                 bottleneck_dim: int = 256):
        super().__init__()
        self.extractor = extractor
        self.feature_processor = feature_processor
        self.in_dim = in_dim
        
        # Identical bottleneck definition to FFMulticlass for easy loading
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        
        # Learnable parameters for scaling cosine to logits
        # Initialize similar to FFCosine: scale=10, bias=-5
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def forward(self, input_data_ground_truth, input_data_tested, label=None):
        # 1. Extract and Process (Sequentially to save memory)
        # Process GT branch immediately to free raw embeddings
        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_gt = self.feature_processor(emb_gt)

        # Process Test branch
        emb_test = self.extractor.extract_features(input_data_tested)
        emb_test = self.feature_processor(emb_test)

        # 2. Bottleneck (Shared Map)
        # Use simple vectors, no activation (linear projection to the map)
        proj_gt = self.bottleneck(emb_gt)
        proj_test = self.bottleneck(emb_test)

        # 3. Compute Cosine Similarity
        cosine = F.cosine_similarity(proj_gt, proj_test, dim=1, eps=1e-8)
        
        # 4. Scale & Bias (Map -1..1 to real logits)
        scale = torch.clamp(self.scale, max=30.0)
        logit = (cosine * scale) + self.bias
        
        # 5. Return Logits and Probs
        # FFPairTrainer expects (logit, prob) or (logits, probs)
        # If we return a single logit, FFPairTrainer might need adjustment or we act like FFCosine
        # FFCosine returns: logits = torch.stack([-logit, logit], dim=1), probs = F.softmax(...)
        
        # Let's match FFCosine's output format for compatibility with FFPairTrainer
        logits = torch.stack([-logit, logit], dim=1) 
        probs = F.softmax(logits, dim=1)
        
        return logits, probs

    def forward_from_embeddings(self, emb_gt, emb_test, label=None):
        """
        Forward pass using precomputed embeddings (after feature_processor).
        """
        proj_gt = self.bottleneck(emb_gt)
        proj_test = self.bottleneck(emb_test)

        cosine = F.cosine_similarity(proj_gt, proj_test, dim=1, eps=1e-8)
        scale = torch.clamp(self.scale, max=30.0)
        logit = (cosine * scale) + self.bias

        logits = torch.stack([-logit, logit], dim=1)
        probs = F.softmax(logits, dim=1)
        return logits, probs
