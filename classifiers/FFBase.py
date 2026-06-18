import torch.nn as nn


class AdaptiveNorm1d(nn.Module):
    """
    Use BatchNorm when batch size is reasonably large; fall back to GroupNorm for tiny batches.
    This keeps small-batch training stable without changing behavior for typical batch sizes.
    """

    def __init__(self, num_features: int, switch_batch_size: int = 8):
        super().__init__()
        self.switch_batch_size = switch_batch_size
        self.bn = nn.BatchNorm1d(num_features)
        self.gn = nn.GroupNorm(1, num_features)

    def forward(self, x):
        batch_size = x.shape[0]
        if batch_size < self.switch_batch_size:
            return self.gn(x)
        return self.bn(x)


class FFBase(nn.Module):
    """
    Base class for feedforward classifiers, inherited by FFConcatBase and FFDiffBase.
    """

    def __init__(
        self,
        extractor,
        feature_processor,
        in_dim=1024,
        num_classes: int = 2,
        embedding_dim: int | None = None,
    ):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier.
        param embedding_dim: Optional penultimate embedding dimension.
                             If None, defaults to in_dim // 4 (original behavior).
        """

        super().__init__()

        self.extractor = extractor
        self.feature_processor = feature_processor
        self.num_classes = num_classes

        # Allow variable input dimension, mainly for base (768 features), large (1024 features) and extra-large (1920 features) models.
        self.layer1_in_dim = in_dim
        self.layer1_out_dim = in_dim // 2
        self.layer2_in_dim = self.layer1_out_dim
        if embedding_dim is None:
            self.layer2_out_dim = self.layer2_in_dim // 2
        else:
            self.layer2_out_dim = int(embedding_dim)
            if self.layer2_out_dim <= 0:
                raise ValueError("embedding_dim must be a positive integer.")

        self.classifier = nn.Sequential(
            nn.Linear(self.layer1_in_dim, self.layer1_out_dim),
            AdaptiveNorm1d(self.layer1_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_in_dim, self.layer2_out_dim),
            AdaptiveNorm1d(self.layer2_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_out_dim, self.num_classes),
        )

    def forward(self, input_gt, input_tested):
        raise NotImplementedError("Forward pass not implemented in the base class.")

    def extract_embedding(self, emb):
        """
        Return the penultimate-layer embedding produced by the classifier stack.
        """
        if not isinstance(self.classifier, nn.Sequential) or len(self.classifier) < 2:
            raise RuntimeError("Classifier stack does not support embedding extraction.")
        for layer in list(self.classifier)[:-1]:
            emb = layer(emb)
        return emb

    def get_param_groups(self):
        """
        Optional optimizer param grouping.

        When SSL extractor fine-tuning is enabled, keep a separate (usually much smaller)
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
