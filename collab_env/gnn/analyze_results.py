#!/usr/bin/env python
"""
Analysis script for GNN training results.
Loads JSON result files and provides comprehensive analysis with plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from collab_env.gnn.analysis_utils import load_results, results_to_dataframe, get_summary_stats

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def create_dataframe_with_summary(results):
    """Create DataFrame and print summary information."""
    # Filter results by status
    successful_results = [r for r in results if r.get("status") == "success"]
    failed_results = [r for r in results if r.get("status") != "success"]
    
    print("\nResults summary:")
    print(f"  Total loaded: {len(results)} results")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    
    if failed_results:
        print("\nFailure reasons:")
        import pandas as pd
        failure_counts = pd.Series([r.get("status", "unknown") for r in failed_results]).value_counts()
        for status, count in failure_counts.items():
            print(f"  {status}: {count}")
    
    # Use utility function to create DataFrame
    df = results_to_dataframe(results)
    
    return df


def plot_loss_distribution(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot distribution of final losses."""
    if df.empty:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Loss Distribution Analysis', fontsize=16)
    
    # Overall loss distribution
    axes[0, 0].hist(df['final_loss'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Final Loss')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Overall Loss Distribution')
    axes[0, 0].set_yscale('log')
    
    # Loss by model type
    if 'model_name' in df.columns:
        model_losses = [df[df['model_name'] == model]['final_loss'].values for model in df['model_name'].unique()]
        axes[0, 1].boxplot(model_losses, labels=df['model_name'].unique())
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Final Loss')
        axes[0, 1].set_title('Loss by Model Type')
        axes[0, 1].set_yscale('log')
    
    # Loss by number of heads
    if 'heads' in df.columns:
        head_losses = [df[df['heads'] == heads]['final_loss'].values for heads in sorted(df['heads'].unique())]
        axes[1, 0].boxplot(head_losses, labels=sorted(df['heads'].unique()))
        axes[1, 0].set_xlabel('Number of Heads')
        axes[1, 0].set_ylabel('Final Loss')
        axes[1, 0].set_title('Loss by Attention Heads')
        axes[1, 0].set_yscale('log')
    
    # Loss by noise level
    if 'noise' in df.columns:
        noise_losses = [df[df['noise'] == noise]['final_loss'].values for noise in sorted(df['noise'].unique())]
        axes[1, 1].boxplot(noise_losses, labels=sorted(df['noise'].unique()))
        axes[1, 1].set_xlabel('Noise Level')
        axes[1, 1].set_ylabel('Final Loss')
        axes[1, 1].set_title('Loss by Noise Level')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_hyperparameter_heatmap(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot heatmap of hyperparameter combinations."""
    if df.empty or len(df) < 2:
        return
    
    # Create pivot table for heatmap
    if 'heads' in df.columns and 'visual_range' in df.columns:
        # Average loss for each combination
        pivot_df = df.groupby(['heads', 'visual_range'])['final_loss'].mean().unstack()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='viridis_r', 
                   cbar_kws={'label': 'Average Final Loss'})
        plt.title('Average Loss by Heads vs Visual Range')
        plt.xlabel('Visual Range')
        plt.ylabel('Number of Heads')
        plt.tight_layout()
        plt.savefig(save_dir / 'hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_seed_variance(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot variance across different seeds for each configuration."""
    if df.empty or 'seed' not in df.columns:
        return
    
    # Group by configuration (excluding seed) and calculate statistics
    config_cols = ['model_name', 'noise', 'heads', 'visual_range']
    available_cols = [col for col in config_cols if col in df.columns]
    
    if not available_cols:
        return
    
    seed_stats = df.groupby(available_cols)['final_loss'].agg(['mean', 'std', 'count']).reset_index()
    seed_stats = seed_stats[seed_stats['count'] > 1]  # Only configs with multiple seeds
    
    if seed_stats.empty:
        print("No configurations with multiple seeds found")
        return
    
    # Create config labels
    seed_stats['config_label'] = seed_stats[available_cols].apply(
        lambda x: '_'.join([f"{col}:{x[col]}" for col in available_cols]), axis=1
    )
    
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(seed_stats))
    
    plt.bar(x_pos, seed_stats['mean'], yerr=seed_stats['std'], capsize=5, alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Final Loss (Mean ± Std)')
    plt.title('Loss Variance Across Seeds')
    plt.xticks(x_pos, seed_stats['config_label'], rotation=45, ha='right')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_dir / 'seed_variance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_timeline(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot training results over time if worker_id is available."""
    if df.empty or 'worker_id' not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Color by model type
    models = df['model_name'].unique() if 'model_name' in df.columns else ['all']
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        if len(models) > 1:
            model_df = df[df['model_name'] == model]
        else:
            model_df = df
        
        plt.scatter(model_df['worker_id'], model_df['final_loss'], 
                   label=model, alpha=0.7, c=[colors[i]])
    
    plt.xlabel('Worker ID (Order of Completion)')
    plt.ylabel('Final Loss')
    plt.title('Training Results by Completion Order')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_stats(df, save_dir: Path) -> None:
    """Generate and save summary statistics using utility function."""
    if df.empty:
        return
    
    # Get summary stats using utility function
    stats = get_summary_stats(df)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    print("\nOverall Results:")
    print(f"  Total successful runs: {stats['total_runs']}")
    print(f"  Best loss: {stats['best_loss']:.6f}")
    print(f"  Worst loss: {stats['worst_loss']:.6f}")
    print(f"  Mean loss: {stats['mean_loss']:.6f}")
    print(f"  Median loss: {stats['median_loss']:.6f}")
    print(f"  Std dev: {stats['std_loss']:.6f}")
    
    # Best configuration
    if 'best_config' in stats:
        best_config = stats['best_config']
        print("\nBest Configuration:")
        for col in ['model_name', 'noise', 'heads', 'visual_range', 'seed']:
            if col in best_config:
                print(f"  {col}: {best_config[col]}")
        print(f"  Final loss: {best_config['final_loss']:.6f}")
    
    # Statistics by model
    if 'by_model' in stats:
        print("\nBy Model:")
        import pandas as pd
        model_stats = pd.DataFrame.from_dict(stats['by_model'], orient='index')
        print(model_stats)
    
    # Statistics by hyperparameters
    for param in ['heads', 'noise', 'visual_range']:
        param_key = f'by_{param}'
        if param_key in stats:
            print(f"\nBy {param}:")
            import pandas as pd
            param_stats = pd.DataFrame.from_dict(stats[param_key], orient='index')
            print(param_stats)
    
    # Save detailed stats to file
    with open(save_dir / 'summary_stats.txt', 'w') as f:
        f.write("GNN Training Results Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total successful runs: {stats['total_runs']}\n")
        f.write(f"Best loss: {stats['best_loss']:.6f}\n")
        f.write(f"Mean loss: {stats['mean_loss']:.6f}\n")
        f.write(f"Std dev: {stats['std_loss']:.6f}\n\n")
        
        if 'best_config' in stats:
            best_config = stats['best_config']
            f.write("Best Configuration:\n")
            for col in ['model_name', 'noise', 'heads', 'visual_range', 'seed']:
                if col in best_config:
                    f.write(f"  {col}: {best_config[col]}\n")
            f.write(f"  Final loss: {best_config['final_loss']:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze GNN training results")
    parser.add_argument("results_path", type=str, 
                       help="Path to results JSON file or directory containing JSON files")
    parser.add_argument("--output-dir", type=str, default="analysis_output",
                       help="Output directory for plots and analysis (default: analysis_output)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from: {args.results_path}")
    results_path = Path(args.results_path)
    
    # Load results
    try:
        results = load_results(results_path)
        print(f"Loaded {len(results)} total results")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    if not results:
        print("No results found!")
        return
    
    # Convert to DataFrame with summary
    df = create_dataframe_with_summary(results)
    
    if df.empty:
        print("No successful results to analyze!")
        return
    
    print(f"\nAnalyzing {len(df)} successful results...")
    print(f"Columns available: {list(df.columns)}")
    
    # Generate analysis
    print("\nGenerating plots...")
    
    plot_loss_distribution(df, output_dir)
    plot_hyperparameter_heatmap(df, output_dir)
    plot_seed_variance(df, output_dir)
    plot_training_timeline(df, output_dir)
    
    # Generate summary statistics
    generate_summary_stats(df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("Check the following files:")
    print("  - loss_distribution.png")
    print("  - hyperparameter_heatmap.png") 
    print("  - seed_variance.png")
    print("  - training_timeline.png")
    print("  - summary_stats.txt")


if __name__ == "__main__":
    main()