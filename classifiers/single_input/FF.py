import torch.nn.functional as F

from classifiers.FFBase import FFBase


class FF(FFBase):
    """
    Feedforward classifier for audio embeddings.
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

        super().__init__(
            extractor,
            feature_processor,
            in_dim=in_dim,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )

    def forward(self, waveforms):
        """
        Forward pass through the model.

        Extract features from the audio data, process them and pass them through the classifier.

        param embeddings: Audio waveforms of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb = self.extractor.extract_features(waveforms)

        emb = self.feature_processor(emb)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob

    def extract_embedding(self, waveforms):
        """
        Return penultimate-layer embeddings for the given waveforms.
        """
        emb = self.extractor.extract_features(waveforms)
        emb = self.feature_processor(emb)
        return super().extract_embedding(emb)
