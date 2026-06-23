import torch
import tonic
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import FullPredictiveVisionPipeline
import os

def plot_qualitative_raster(raw, expected, sparse, sample_idx=0):
    raw_sample = raw[sample_idx, 0].detach()
    expected_sample = expected[sample_idx, 0].detach()
    sparse_sample = sparse[sample_idx, 0].detach()

    raw_2d = raw_sample.sum(dim=0)
    expected_2d = expected_sample.sum(dim=0)
    sparse_2d = sparse_sample.sum(dim=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Event-Driven Predictive Vision: Spatial Sparsity", fontsize=16, fontweight='bold')

    axes[0].imshow(raw_2d, cmap='hot', interpolation='nearest')
    axes[0].set_title("1. Raw DVS Input (All Change)")
    axes[0].axis('off')

    axes[1].imshow(expected_2d, cmap='hot', interpolation='nearest')
    axes[1].set_title("2. Model Expectation (Predictable)")
    axes[1].axis('off')

    axes[2].imshow(sparse_2d, cmap='hot', interpolation='nearest')
    axes[2].set_title("3. Final Output (Informative Surprise)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("figure_2_raster_plot.png", dpi=300)
    print("Saved 'figure_2_raster_plot.png'")

def plot_quantitative_density(raw, layer1, sparse):
    # CRITICAL FIX: Strict binary active-voxel ratio mapping
    densities = [
        (raw != 0).float().mean().item(),
        (layer1 != 0).float().mean().item(),
        (sparse != 0).float().mean().item()
    ]
    labels = ['Raw Input\n(O(k) events)', 'Noise Gate\n(Layer 1)', 'Sparse Output\n(Layer 4)']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, densities, color=['#e63946', '#f4a261', '#2a9d8f'])
    
    plt.title("Progressive Signal Sparsification", fontsize=14, fontweight='bold')
    plt.ylabel("Event Density (Active Voxel Ratio)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, 
                 f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("figure_3_density_reduction.png", dpi=300)
    print("Saved 'figure_3_density_reduction.png'")

def main():
    print("Loading N-Caltech101 Data...")
    
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
    
    frames, _ = next(iter(dataloader))
    # CRITICAL FIX: Sum polarities properly
    frames = frames.sum(dim=2, keepdim=True).permute(0, 2, 1, 3, 4).float()
    
    print("Processing Pipeline...")
    metrics = pipeline(frames)
    
    plot_qualitative_raster(frames, metrics["expected"], metrics["output"])
    plot_quantitative_density(frames, metrics["cleaned"], metrics["output"])
    
    print("\nVisualizations complete! Check your project folder for the PNG files.")

if __name__ == "__main__":
    main()