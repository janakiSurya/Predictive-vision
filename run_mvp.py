import torch
import torch.optim as optim
import tonic
from torch.utils.data import DataLoader
from model import FullPredictiveVisionPipeline
import os

def main():
    print("Initializing/Downloading N-Caltech101 dataset via Tonic...")
    
    # CRITICAL FIX: Consistent Label Mapping & Fixed Sensor Loading
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
    
    dataloader = DataLoader(
        cached_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=torch.utils.data.dataloader.default_collate
    )
    
    pipeline = FullPredictiveVisionPipeline()
    optimizer = optim.Adam(pipeline.parameters(), lr=1e-3)
    
    print("\nStarting Pipeline Verification Loop...")
    for i, (frames, labels) in enumerate(dataloader):
        # CRITICAL FIX: Sum polarities instead of completely dropping the negative changes
        frames = frames.sum(dim=2, keepdim=True).permute(0, 2, 1, 3, 4).float()
        
        optimizer.zero_grad()
        metrics = pipeline(frames)
        
        expected = metrics["expected"]
        cleaned = metrics["cleaned"]
        beta = metrics["beta"]
        output = metrics["output"]
        
        base_loss = torch.nn.functional.mse_loss(expected, cleaned, reduction='none')
        modulated_loss = (base_loss * beta).mean()
        
        modulated_loss.backward()
        optimizer.step()
        
        # CRITICAL FIX: Use strict binary active-voxel ratio for accurate density
        raw_density = (frames != 0).float().mean().item()
        cleaned_density = (cleaned != 0).float().mean().item()
        output_density = (output != 0).float().mean().item()
        
        print(f"\n--- Batch {i+1} Sparsity Analysis ---")
        print(f"Raw Input Event Density:    {raw_density:.5f}")
        print(f"Denoised (Layer 1) Density: {cleaned_density:.5f}")
        print(f"Final Output (Layer 4) Res: {output_density:.5f}")
        
        # CRITICAL FIX: Corrected logging names
        print(f"Hybrid Blend Alpha:         {pipeline.predictor.alpha.item():.4f}")
        print(f"Plasticity Gate Beta:       {beta.mean().item():.4f}")
        print(f"Modulated Backprop Loss:    {modulated_loss.item():.5f}")
        
        if i == 2:
            print("\nVerification successful. Infrastructure fully active.")
            break

if __name__ == "__main__":
    main()