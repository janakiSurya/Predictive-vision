import torch
import torch.nn as nn
import torch.optim as optim
import tonic
import os
from torch.utils.data import DataLoader, random_split
from model import FullPredictiveVisionPipeline

# ---------------------------------------------------------
# 1. The Downstream Task 
# ---------------------------------------------------------
class EventClassifier(nn.Module):
    def __init__(self, num_classes=101): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)) 
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------------------------------------------------------
# 2. Rigorous Evaluation Loop (Train + Validation splits)
# ---------------------------------------------------------
def evaluate_pipeline(train_loader, test_loader, epochs=5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nRunning Full Empirical Benchmark on: {device}")

    # Initialize Models
    predictive_pipeline = FullPredictiveVisionPipeline().to(device)
    baseline_classifier = EventClassifier(num_classes=101).to(device)
    sparse_classifier = EventClassifier(num_classes=101).to(device)
    
    # Optimizers (CRITICAL FIX: We are now training the pipeline too!)
    opt_pipeline = optim.Adam(predictive_pipeline.parameters(), lr=0.001)
    opt_baseline = optim.Adam(baseline_classifier.parameters(), lr=0.001)
    opt_sparse = optim.Adam(sparse_classifier.parameters(), lr=0.001)
    
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss(reduction='none')

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        predictive_pipeline.train()
        baseline_classifier.train()
        sparse_classifier.train()
        
        print(f"\n--- EPOCH {epoch+1}: TRAINING ---")
        for i, (frames, labels) in enumerate(train_loader):
            # CRITICAL FIX: Sum polarities instead of dropping channel 1
            # Tonic format: [B, T, C, H, W] -> Our format: [B, C, T, H, W]
            raw_events = frames.sum(dim=2, keepdim=True).permute(0, 2, 1, 3, 4).float().to(device)
            labels = labels.to(device).long()
            
            # 1. Train Baseline
            opt_baseline.zero_grad()
            raw_logits = baseline_classifier(raw_events)
            loss_base = criterion(raw_logits, labels)
            loss_base.backward()
            opt_baseline.step()

            # 2. Train Pipeline & Sparse Classifier
            opt_pipeline.zero_grad()
            opt_sparse.zero_grad()
            
            pipeline_out = predictive_pipeline(raw_events)
            sparse_events = pipeline_out["output"]
            
            # Internal Pipeline Loss (Learn to predict)
            expected = pipeline_out["expected"]
            cleaned = pipeline_out["cleaned"]
            beta = pipeline_out["beta"]
            p_loss = (mse_loss(expected, cleaned) * beta).mean()
            
            # Downstream Task Loss
            sparse_logits = sparse_classifier(sparse_events)
            loss_sparse = criterion(sparse_logits, labels)
            
            # Joint optimization
            total_loss = p_loss + loss_sparse
            total_loss.backward()
            opt_pipeline.step()
            opt_sparse.step()

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(train_loader)} training batches...")

        # --- VALIDATION PHASE (CRITICAL FIX: Unseen data only) ---
        predictive_pipeline.eval()
        baseline_classifier.eval()
        sparse_classifier.eval()
        
        correct_raw, correct_sparse, total_samples = 0, 0, 0
        total_raw_density, total_sparse_density = 0.0, 0.0
        
        print(f"--- EPOCH {epoch+1}: VALIDATION ---")
        with torch.no_grad():
            for frames, labels in test_loader:
                raw_events = frames.sum(dim=2, keepdim=True).permute(0, 2, 1, 3, 4).float().to(device)
                labels = labels.to(device).long()
                
                pipeline_out = predictive_pipeline(raw_events)
                sparse_events = pipeline_out["output"]
                
                # CRITICAL FIX: Consistent density calculation
                total_raw_density += (raw_events != 0).float().mean().item()
                total_sparse_density += (sparse_events != 0).float().mean().item()
                
                raw_logits = baseline_classifier(raw_events)
                sparse_logits = sparse_classifier(sparse_events)
                
                correct_raw += (raw_logits.argmax(dim=1) == labels).sum().item()
                correct_sparse += (sparse_logits.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)

        # Validation Metrics
        acc_raw = (correct_raw / total_samples) * 100
        acc_sparse = (correct_sparse / total_samples) * 100
        avg_raw_density = total_raw_density / len(test_loader)
        avg_sparse_density = total_sparse_density / len(test_loader)
        reduction = (1.0 - (avg_sparse_density / (avg_raw_density + 1e-9))) * 100

        print(f"=== EPOCH {epoch+1} VALIDATION SUMMARY ===")
        print(f"  Data Reduction:    {reduction:.2f}% (Raw: {avg_raw_density:.4f} -> Sparse: {avg_sparse_density:.4f})")
        print(f"  Baseline Accuracy: {acc_raw:.2f}%")
        print(f"  Sparse Accuracy:   {acc_sparse:.2f}%\n")

# ---------------------------------------------------------
# 3. Safe Data Orchestration
# ---------------------------------------------------------
if __name__ == "__main__":
    # CLEAR CACHE CRASH PREVENTION
    cache_dir = './cache'
    if os.path.exists(cache_dir):
        print("Clearing old dataset cache to prevent dimension/label crashes...")
        import shutil
        shutil.rmtree(cache_dir)
        
    print("Loading N-Caltech101 Data via Tonic...")
    
    # Pre-build label map safely
    temp_dataset = tonic.datasets.NCALTECH101(save_to='./data')
    unique_classes = sorted(list(set([t[1] for t in temp_dataset])))
    class_map = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    
    def safe_label_map(target):
        if isinstance(target, bytes):
            target = target.decode('utf-8')
        return class_map[target]
    
    dataset = tonic.datasets.NCALTECH101(save_to='./data', target_transform=safe_label_map)
    
    fixed_sensor_size = (304, 240, 2)
    to_frame = tonic.transforms.ToFrame(sensor_size=fixed_sensor_size, n_time_bins=10)
    cached_dataset = tonic.DiskCachedDataset(dataset, transform=to_frame, cache_path='./cache')
    
    # CRITICAL FIX: Train / Test Split (80% Train, 20% Test)
    train_size = int(0.8 * len(cached_dataset))
    test_size = len(cached_dataset) - train_size
    train_dataset, test_dataset = random_split(cached_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=torch.utils.data.dataloader.default_collate)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=torch.utils.data.dataloader.default_collate)
    
    evaluate_pipeline(train_loader, test_loader, epochs=5)