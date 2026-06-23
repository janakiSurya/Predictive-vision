import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseGate(nn.Module):
    def __init__(self, threshold=3.0):
        super().__init__()
        self.threshold = threshold
        self.support_kernel = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.support_kernel.weight, 1.0)
        for param in self.support_kernel.parameters():
            param.requires_grad = False

    def forward(self, x):
        local_density = self.support_kernel(x)
        mask = (local_density >= self.threshold).float()
        return x * mask

class PhysicsPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        prediction = torch.zeros_like(x)
        prediction[:, :, 1:, :, :] = x[:, :, :-1, :, :]
        return prediction

class LearnedPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            # CRITICAL FIX: Changed from Sigmoid to ReLU. Event sums can exceed 1.0; 
            # Sigmoid was artificially clamping the network's predictive power.
            nn.ReLU() 
        )

    def forward(self, x):
        return self.net(x)

class HybridPredictor(nn.Module):
    def __init__(self, initial_alpha=0.9):
        super().__init__()
        self.physics_model = PhysicsPredictor()
        self.learned_model = LearnedPredictor()
        self.alpha = nn.Parameter(torch.tensor([initial_alpha]))

    def forward(self, x):
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        pred_phys = self.physics_model(x)
        pred_learn = self.learned_model(x)
        return (alpha_clamped * pred_phys) + ((1.0 - alpha_clamped) * pred_learn)

class PlasticityController(nn.Module):
    def __init__(self, temporal_window=3):
        super().__init__()
        self.temporal_window = temporal_window
        self.coherence_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, residual):
        padding = self.temporal_window - 1
        padded_res = F.pad(residual, (0, 0, 0, 0, padding, 0))
        persistence = F.avg_pool3d(padded_res, kernel_size=(self.temporal_window, 1, 1), stride=1)
        
        B, C, T, H, W = residual.shape
        res_reshaped = residual.view(B * C * T, 1, H, W)
        coherence = self.coherence_pool(torch.abs(res_reshaped))
        coherence = coherence.view(B, C, T, H, W)
        
        beta = torch.sigmoid((persistence * coherence) * 10.0) 
        return beta

class ErrorExtractor(nn.Module):
    def __init__(self, error_threshold=0.2):
        super().__init__()
        self.error_threshold = error_threshold

    def forward(self, observed, predicted):
        residual = observed - predicted
        mask = (torch.abs(residual) >= self.error_threshold).float()
        return residual * mask

class RegimeClassifier(nn.Module):
    def __init__(self, num_regimes=4):
        super().__init__()
        self.num_regimes = num_regimes
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.GRU(input_size=1, hidden_size=8, batch_first=True)
        self.classifier = nn.Linear(8, num_regimes)

    def forward(self, thresholded_error):
        B, C, T, H, W = thresholded_error.shape
        error_reshaped = thresholded_error.view(B * T, C, H, W)
        global_features = self.spatial_pool(torch.abs(error_reshaped)).view(B, T, 1)
        
        rnn_out, _ = self.rnn(global_features)
        regime_logits = self.classifier(rnn_out)
        return F.softmax(regime_logits, dim=-1)

class SparseOutput(nn.Module):
    # CRITICAL FIX: Synchronized default threshold to 0.2
    def __init__(self, confidence_threshold=0.2): 
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(self, thresholded_error, regime_probs):
        max_probs, _ = torch.max(regime_probs, dim=-1)
        confidence_mask = (max_probs >= self.confidence_threshold).float()
        confidence_mask = confidence_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return thresholded_error * confidence_mask

class FullPredictiveVisionPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_gate = NoiseGate(threshold=2.0)
        self.predictor = HybridPredictor(initial_alpha=0.8)
        self.error_extractor = ErrorExtractor(error_threshold=0.2)
        self.plasticity_ctrl = PlasticityController()
        self.regime_classifier = RegimeClassifier(num_regimes=4)
        self.sparse_output = SparseOutput(confidence_threshold=0.2)

    def forward(self, raw_events):
        cleaned_events = self.noise_gate(raw_events)
        expected_events = self.predictor(cleaned_events)
        raw_residual = self.error_extractor(cleaned_events, expected_events)
        
        beta_gate = self.plasticity_ctrl(raw_residual)
        regime_probs = self.regime_classifier(raw_residual)
        final_sparse_signal = self.sparse_output(raw_residual, regime_probs)
        
        return {
            "output": final_sparse_signal,
            "beta": beta_gate,
            "regime": regime_probs,
            "cleaned": cleaned_events,
            "expected": expected_events
        }