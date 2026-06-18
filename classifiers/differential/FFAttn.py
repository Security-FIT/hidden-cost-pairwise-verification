import torch
import torch.nn as nn
import torch.nn.functional as F

from classifiers.FFBase import FFBase

#### IMPORTANT NOTE! ####
# Using the attn_map (or attn_output), i.e. the first value from nn.MultiHeadAttention does NOT work
# as a direct input to the processors (AASIST, MHFA), not even with a residual connection. Suspect this might
# be due to the specific way these processors are designed to handle input features, expecting strictly
# spectro-temporal representations, which breaks when using the attention outputs, as those are completely
# different, more like a similarity. In the future, this approach could be viable if the processors are
# adapted to handle such representations or we come up with an alternative way of processing the attn_map.

class FFAttnBase(FFBase):
    """
    Base class for feedforward classifiers which use cross-attention
    between the test and reference recording for classification.
    """

    def __init__(self, extractor, feature_processor, in_dim=1024, head_nb=8):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        """

        super().__init__(extractor, feature_processor, in_dim)
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=head_nb)

        # Shared attention gain (scaled by attn_cap) to keep fusion subtle.
        # For AASIST we are extra conservative; for other processors we allow a bit more.
        processor_name = type(self.feature_processor).__name__
        if processor_name == "AASIST":
            # Start with a very small gain for AASIST; let training increase it if useful.
            self.attn_gain = nn.Parameter(torch.tensor(-2.0))
            self.attn_cap = 0.1
        else:
            self.attn_gain = nn.Parameter(torch.tensor(-1.0))
            self.attn_cap = 0.5

        # Light normalization after fusion to stabilise inputs to AASIST/MHFA/etc.
        self.post_fusion_ln = nn.LayerNorm(in_dim)

        # Cosine-similarity scoring with learnable scale/bias (shared by attn variants)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    @property
    def _is_aasist(self) -> bool:
        return type(self.feature_processor).__name__ == "AASIST"


class FFAttn1(FFAttnBase):
    """
    Feedforward classifier with cross-attention between test and reference recordings.
    """

    def __init__(self, extractor, feature_processor, in_dim=1024):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        """
        super().__init__(extractor, feature_processor, in_dim)

        # Soft blending factor for attenuation: gate = (1 - beta) + beta * gate_raw
        self.atten_beta = nn.Parameter(torch.tensor(-2.0))

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass of the model.

        Extract features from the test and reference data, compute cross-attention
        between the embeddings, enhance the test embedding with the cross-attention, 
        process the informed embedding and pass it to the classifier.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        # Reshape so that we compute cross-attention per transformer layer
        layers, batches, time_gt, feature = emb_gt.shape
        time_test = emb_test.shape[2]

        # and transpose to have time as the first dimension (as expected by nn.MultiheadAttention)
        stacked_gt = emb_gt.view(layers * batches, time_gt, feature).transpose(
            0, 1
        )  # (time_gt, layers * batches, feature)
        stacked_test = emb_test.view(layers * batches, time_test, feature).transpose(
            0, 1
        )  # (time_test, layers * batches, feature)

        # Compute the attention map, self.attn(query (from test), key (from reference), value (from reference))
        # Note: the attention map is computed for each layer and batch, so we have to reshape it back
        # attn_map = (time_test, layers * batches, feature)
        attn_map_test, attn_weights_test = self.attn(stacked_test, stacked_gt, stacked_gt)
        attn_weights_test = attn_weights_test.view(
            layers, batches, time_test, time_gt
        )

        attn_map_gt, attn_weights_gt = self.attn(stacked_gt, stacked_test, stacked_test)
        attn_weights_gt = attn_weights_gt.view(
            layers, batches, time_gt, time_test
        )

        gate_test = attn_weights_test.max(dim=3).values  # [L, B, T_test]
        gate_test = torch.sigmoid(gate_test)  # normalize to [0, 1]

        gate_gt = attn_weights_gt.max(dim=3).values  # [L, B, T_gt]
        gate_gt = torch.sigmoid(gate_gt)  # normalize to [0, 1]

        # Soften attenuation: blend gate with identity using a learnable beta
        beta = torch.sigmoid(self.atten_beta)  # in (0,1)
        gate_test = (1 - beta) + beta * gate_test
        gate_gt = (1 - beta) + beta * gate_gt

        # Frame-weighting fusion; for AASIST, only affect the last SSL layer
        if self._is_aasist:
            test_emb = emb_test.clone()
            gt_emb = emb_gt.clone()
            test_emb[-1] = emb_test[-1] * gate_test[-1].unsqueeze(-1)
            gt_emb[-1] = emb_gt[-1] * gate_gt[-1].unsqueeze(-1)
        else:
            test_emb = emb_test * gate_test.unsqueeze(-1)
            gt_emb = emb_gt * gate_gt.unsqueeze(-1)

        # Post-fusion normalization to stabilise processor input
        test_emb = self.post_fusion_ln(test_emb)
        gt_emb = self.post_fusion_ln(gt_emb)

        # Process the features
        emb_test = self.feature_processor(test_emb)
        emb_gt = self.feature_processor(gt_emb)

        # 2. Compute Cosine Similarity (-1 to 1)
        similarity = F.cosine_similarity(emb_gt, emb_test, dim=1, eps=1e-8)

        # 3. Apply Affine Transformation (Scale & Shift)
        logit = (similarity * self.scale) + self.bias

        # 4. Convert to 2-class logits and probabilities for CrossEntropyLoss
        logits = torch.stack([-logit, logit], dim=1) 
        
        probs = F.softmax(logits, dim=1)

        return logits, probs

        # out = self.classifier(emb)
        # prob = F.softmax(out, dim=1)

        # return out, prob
    

class FFAttn2(FFAttnBase):
    """
    Feedforward classifier with cross-attention between test and reference recordings.
    """

    def __init__(self, extractor, feature_processor, in_dim=1024):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        """
        super().__init__(extractor, feature_processor, in_dim)

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass of the model.

        Extract features from the test and reference data, compute cross-attention
        between the embeddings, enhance both embeddings with a gated residual,
        process them, and score with cosine similarity (like FFAttn1).

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        # Reshape so that we compute cross-attention per transformer layer
        layers, batches, time_gt, feature = emb_gt.shape
        time_test = emb_test.shape[2]

        # and transpose to have time as the first dimension (as expected by nn.MultiheadAttention)
        stacked_gt = emb_gt.view(layers * batches, time_gt, feature).transpose(
            0, 1
        )  # (time_gt, layers * batches, feature)
        stacked_test = emb_test.view(layers * batches, time_test, feature).transpose(
            0, 1
        )  # (time_test, layers * batches, feature)

        # Compute cross-attention weights (test -> gt and gt -> test)
        _, attn_weights_test = self.attn(stacked_test, stacked_gt, stacked_gt)
        attn_weights_test = attn_weights_test.view(layers, batches, time_test, time_gt)

        _, attn_weights_gt = self.attn(stacked_gt, stacked_test, stacked_test)
        attn_weights_gt = attn_weights_gt.view(layers, batches, time_gt, time_test)

        # Use how focused the attention is as a gate (frame weighting)
        gate_test = torch.sigmoid(attn_weights_test.max(dim=3).values)  # [L, B, T_test]
        gate_gt = torch.sigmoid(attn_weights_gt.max(dim=3).values)      # [L, B, T_gt]

        # Keep residual small and learnable; cap to avoid swamping AASIST/MHFA
        gain = self.attn_cap * torch.sigmoid(self.attn_gain)

        if self._is_aasist:
            # For AASIST, only fuse the last SSL layer; keep others untouched.
            test_emb = emb_test.clone()
            gt_emb = emb_gt.clone()
            test_emb[-1] = emb_test[-1] + gain * (emb_test[-1] * gate_test[-1].unsqueeze(-1))
            gt_emb[-1] = emb_gt[-1] + gain * (emb_gt[-1] * gate_gt[-1].unsqueeze(-1))
        else:
            # For other processors, allow fusion on all layers.
            test_emb = emb_test + gain * (emb_test * gate_test.unsqueeze(-1))
            gt_emb = emb_gt + gain * (emb_gt * gate_gt.unsqueeze(-1))

        # Post-fusion normalization to stabilise the processor input
        test_emb = self.post_fusion_ln(test_emb)
        gt_emb = self.post_fusion_ln(gt_emb)

        # Process both streams
        emb_test_proc = self.feature_processor(test_emb)
        emb_gt_proc = self.feature_processor(gt_emb)

        # Cosine scoring (FFCosine-style)
        similarity = F.cosine_similarity(emb_gt_proc, emb_test_proc, dim=1, eps=1e-8)
        logit = (similarity * self.scale) + self.bias
        logits = torch.stack([-logit, logit], dim=1)
        prob = F.softmax(logits, dim=1)

        return logits, prob
    

class FFAttn3(FFAttnBase):
    """
    Feedforward classifier with cross-attention between test and reference recordings.
    """

    def __init__(self, extractor, feature_processor, in_dim=1024):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        """
        super().__init__(extractor, feature_processor, in_dim)

        self.attention_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(),
            nn.Linear(in_dim * 4, in_dim),
            nn.LayerNorm(in_dim)
        )

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass of the model.

        Extract features from the test and reference data, compute cross-attention
        between the embeddings, enhance both embeddings with the cross-attention using residual connection,
        process them and score with the cosine head.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        # Reshape so that we compute cross-attention per transformer layer
        layers, batches, time_gt, feature = emb_gt.shape
        time_test = emb_test.shape[2]

        # and transpose to have time as the first dimension (as expected by nn.MultiheadAttention)
        stacked_gt = emb_gt.view(layers * batches, time_gt, feature).transpose(
            0, 1
        )  # (time_gt, layers * batches, feature)
        stacked_test = emb_test.view(layers * batches, time_test, feature).transpose(
            0, 1
        )  # (time_test, layers * batches, feature)

        attn_map_test, _ = self.attn(stacked_test, stacked_gt, stacked_gt)
        attn_map_gt, _ = self.attn(stacked_gt, stacked_test, stacked_test)

        mlp_aw_test = self.attention_mlp(attn_map_test)  # (T_test, layers * batches, feature)
        mlp_aw_gt = self.attention_mlp(attn_map_gt)      # (T_gt, layers * batches, feature)

        gate_test = torch.softmax(mlp_aw_test, dim=2).transpose(0, 1).view(
            layers, batches, time_test, feature
        )
        gate_gt = torch.softmax(mlp_aw_gt, dim=2).transpose(0, 1).view(
            layers, batches, time_gt, feature
        )

        # Apply the gate to both embeddings with residual; keep gain small and learnable
        gain = self.attn_cap * torch.sigmoid(self.attn_gain)
        if self._is_aasist:
            # AASIST: only fuse the last SSL layer.
            test_emb = emb_test.clone()
            gt_emb = emb_gt.clone()
            test_emb[-1] = emb_test[-1] + gain * (emb_test[-1] * gate_test[-1])
            gt_emb[-1] = emb_gt[-1] + gain * (emb_gt[-1] * gate_gt[-1])
        else:
            test_emb = emb_test + gain * (emb_test * gate_test)  # [L, B, T_test, feature]
            gt_emb = emb_gt + gain * (emb_gt * gate_gt)          # [L, B, T_gt, feature]
        
        # Post-fusion normalization
        test_emb = self.post_fusion_ln(test_emb)
        gt_emb = self.post_fusion_ln(gt_emb)

        # Process the features
        emb_test_proc = self.feature_processor(test_emb)
        emb_gt_proc = self.feature_processor(gt_emb)

        # Cosine-similarity scoring (pairwise head)
        similarity = F.cosine_similarity(emb_gt_proc, emb_test_proc, dim=1, eps=1e-8)
        logit = (similarity * self.scale) + self.bias
        logits = torch.stack([-logit, logit], dim=1)
        prob = F.softmax(logits, dim=1)

        return logits, prob


class FFAttn4(FFAttnBase):
    """
    Feedforward classifier with cross-attention between test and reference recordings.
    """

    def __init__(self, extractor, feature_processor, in_dim=1024):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        """
        super().__init__(extractor, feature_processor, in_dim)

        self.attention_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(),
            nn.Linear(in_dim * 4, in_dim),
            nn.LayerNorm(in_dim)
        )

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass of the model.

        Extract features from the test and reference data, compute cross-attention
        between the embeddings, enhance both embeddings with the cross-attention using residual connection,
        process them and score with the cosine head.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        # Reshape so that we compute cross-attention per transformer layer
        layers, batches, time_gt, feature = emb_gt.shape
        time_test = emb_test.shape[2]

        # and transpose to have time as the first dimension (as expected by nn.MultiheadAttention)
        stacked_gt = emb_gt.view(layers * batches, time_gt, feature).transpose(
            0, 1
        )  # (time_gt, layers * batches, feature)
        stacked_test = emb_test.view(layers * batches, time_test, feature).transpose(
            0, 1
        )  # (time_test, layers * batches, feature)

        attn_map_test, _ = self.attn(stacked_test, stacked_gt, stacked_gt)
        attn_map_gt, _ = self.attn(stacked_gt, stacked_test, stacked_test)

        mlp_aw_test = self.attention_mlp(attn_map_test)  # (T_test, layers * batches, feature)
        mlp_aw_gt = self.attention_mlp(attn_map_gt)      # (T_gt, layers * batches, feature)

        gate_test = mlp_aw_test.transpose(0, 1).view(layers, batches, time_test, feature)
        gate_gt = mlp_aw_gt.transpose(0, 1).view(layers, batches, time_gt, feature)

        # Apply the gate to both embeddings with residual; keep gain small and learnable
        gain = self.attn_cap * torch.sigmoid(self.attn_gain)
        if self._is_aasist:
            test_emb = emb_test.clone()
            gt_emb = emb_gt.clone()
            test_emb[-1] = emb_test[-1] + gain * (emb_test[-1] * gate_test[-1])
            gt_emb[-1] = emb_gt[-1] + gain * (emb_gt[-1] * gate_gt[-1])
        else:
            test_emb = emb_test + gain * (emb_test * gate_test)  # [L, B, T_test, feature]
            gt_emb = emb_gt + gain * (emb_gt * gate_gt)          # [L, B, T_gt, feature]
        
        # Post-fusion normalization
        test_emb = self.post_fusion_ln(test_emb)
        gt_emb = self.post_fusion_ln(gt_emb)

        # Process the features
        emb_test_proc = self.feature_processor(test_emb)
        emb_gt_proc = self.feature_processor(gt_emb)

        # Cosine-similarity scoring (pairwise head)
        similarity = F.cosine_similarity(emb_gt_proc, emb_test_proc, dim=1, eps=1e-8)
        logit = (similarity * self.scale) + self.bias
        logits = torch.stack([-logit, logit], dim=1)
        prob = F.softmax(logits, dim=1)

        return logits, prob


class FFAttn5(nn.Module):
    """
    FFAttn5:
    Cross-attention over latent feature bands of the per-utterance embeddings.

    1) Extract SSL features and pool with the processor to get h_A, h_B ∈ [B, D].
    2) Reshape each into N band tokens of size D / N: [B, N, d_band].
    3) Run cross-attention between bands of A and bands of B.
    4) Pool the attended bands and form a pair embedding from pooled_A, pooled_B,
       |pooled_A - pooled_B| and pooled_A ⊙ pooled_B.
    5) Classify with a small MLP head.

    This design operates purely at the embedding level and is therefore compatible
    with both AASIST and MHFA (and other processors that output a single vector).
    """

    def __init__(self, extractor, feature_processor, in_dim: int = 1024, head_nb: int = 4, n_tokens: int = 16):
        super().__init__()
        self.extractor = extractor
        self.feature_processor = feature_processor
        self.in_dim = in_dim
        self.n_tokens = n_tokens

        if in_dim % n_tokens != 0:
            raise ValueError(f"in_dim ({in_dim}) must be divisible by n_tokens ({n_tokens}) for FFAttn5.")

        self.band_dim = in_dim // n_tokens

        # Cross-attention over feature-band tokens
        self.attn = nn.MultiheadAttention(embed_dim=self.band_dim, num_heads=head_nb, batch_first=True)

        # Pair embedding made from pooled bands of A and B:
        # [pooled_A, pooled_B, |Δ|, ⊙] ⇒ 4 * band_dim
        pair_dim = 4 * self.band_dim
        hidden_dim1 = max(128, pair_dim * 2)
        hidden_dim2 = max(64, hidden_dim1 // 2)

        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 2),
        )

    def forward(self, input_data_ground_truth, input_data_tested):
        # 1) Extract SSL features and pool with the processor
        emb_gt_seq = self.extractor.extract_features(input_data_ground_truth)   # [L, B, T, D_ssl]
        emb_test_seq = self.extractor.extract_features(input_data_tested)       # [L, B, T, D_ssl]

        emb_gt = self.feature_processor(emb_gt_seq)      # [B, in_dim]
        emb_test = self.feature_processor(emb_test_seq)  # [B, in_dim]

        batch_size = emb_gt.size(0)

        # 2) Reshape into feature-band tokens: [B, N, band_dim]
        bands_gt = emb_gt.view(batch_size, self.n_tokens, self.band_dim)
        bands_test = emb_test.view(batch_size, self.n_tokens, self.band_dim)

        # 3) Cross-attention between bands (A as reference, B as query, and vice versa)
        # Shape: [B, N, band_dim]
        attn_test, _ = self.attn(bands_test, bands_gt, bands_gt)  # test attends to gt
        attn_gt, _ = self.attn(bands_gt, bands_test, bands_test)  # gt attends to test

        # 4) Pool attended bands (mean over tokens)
        pooled_test = attn_test.mean(dim=1)  # [B, band_dim]
        pooled_gt = attn_gt.mean(dim=1)      # [B, band_dim]

        diff_abs = torch.abs(pooled_gt - pooled_test)
        prod = pooled_gt * pooled_test

        pair_emb = torch.cat([pooled_gt, pooled_test, diff_abs, prod], dim=1)  # [B, 4 * band_dim]

        # 5) Classify
        logits = self.mlp(pair_emb)
        prob = F.softmax(logits, dim=1)

        return logits, prob

    def forward_from_embeddings(self, emb_gt, emb_test, label=None):
        """
        Forward pass using precomputed embeddings (after feature_processor).
        """
        batch_size = emb_gt.size(0)

        bands_gt = emb_gt.view(batch_size, self.n_tokens, self.band_dim)
        bands_test = emb_test.view(batch_size, self.n_tokens, self.band_dim)

        attn_test, _ = self.attn(bands_test, bands_gt, bands_gt)
        attn_gt, _ = self.attn(bands_gt, bands_test, bands_test)

        pooled_test = attn_test.mean(dim=1)
        pooled_gt = attn_gt.mean(dim=1)

        diff_abs = torch.abs(pooled_gt - pooled_test)
        prod = pooled_gt * pooled_test

        pair_emb = torch.cat([pooled_gt, pooled_test, diff_abs, prod], dim=1)
        logits = self.mlp(pair_emb)
        prob = F.softmax(logits, dim=1)
        return logits, prob
