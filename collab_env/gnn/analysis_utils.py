#!/usr/bin/env python
"""
Simple analysis utilities for loading GNN training results.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file(s).
    
    Args:
        results_path: Path to a JSON file or directory containing JSON files
        
    Returns:
        List of result dictionaries
    """
    results_path = Path(results_path)
    
    if results_path.is_file():
        # Single file
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    elif results_path.is_dir():
        # Directory - load all JSON files
        all_results = []
        json_files = list(results_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    results = json.load(f)
                    # Add metadata about which file this came from
                    for result in results:
                        result['source_file'] = json_file.name
                    all_results.extend(results)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return all_results
    else:
        raise FileNotFoundError(f"Path not found: {results_path}")


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to pandas DataFrame, filtering successful runs.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame with successful results
    """
    # Filter successful results
    successful_results = [r for r in results if r.get("status") == "success"]
    
    if not successful_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(successful_results)
    
    # Convert numeric columns
    numeric_cols = ['noise', 'heads', 'visual_range', 'seed', 'final_loss', 
                   'gpu_id', 'worker_id', 'best_epoch', 'total_epochs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create combined identifier for easier grouping
    if all(col in df.columns for col in ['model_name', 'noise', 'heads', 'visual_range']):
        df['config'] = (df['model_name'] + '_n' + df['noise'].astype(str) + 
                       '_h' + df['heads'].astype(str) + '_vr' + df['visual_range'].astype(str))
    
    return df


def results_to_loss_dict(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Extract losses into nested dictionary organized by configuration.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dict with structure: {config_name: {seed: final_loss}}
    """
    loss_dict = {}
    
    for r in results:
        if r.get("status") != "success":
            continue
            
        # Create config key
        config_key = f"{r['model_name']}_n{r['noise']}_h{r['heads']}_vr{r['visual_range']}"
        
        if config_key not in loss_dict:
            loss_dict[config_key] = {}
        
        # Use seed as sub-key
        seed_key = f"seed_{r['seed']}"
        loss_dict[config_key][seed_key] = r['final_loss']
    
    return loss_dict


def load_and_process(results_path: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Convenience function to load results and return both DataFrame and loss dict.
    
    Args:
        results_path: Path to results JSON file or directory
        
    Returns:
        Tuple of (DataFrame, loss_dict)
    """
    results = load_results(Path(results_path))
    df = results_to_dataframe(results)
    loss_dict = results_to_loss_dict(results)
    
    return df, loss_dict


def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics from results DataFrame.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {}
    
    stats = {
        'total_runs': len(df),
        'best_loss': df['final_loss'].min(),
        'worst_loss': df['final_loss'].max(),
        'mean_loss': df['final_loss'].mean(),
        'median_loss': df['final_loss'].median(),
        'std_loss': df['final_loss'].std(),
    }
    
    # Best configuration
    if not df.empty:
        best_idx = df['final_loss'].idxmin()
        best_config = df.loc[best_idx].to_dict()
        stats['best_config'] = best_config
    
    # Stats by model if available
    if 'model_name' in df.columns:
        model_stats = df.groupby('model_name')['final_loss'].agg(['count', 'mean', 'std', 'min'])
        stats['by_model'] = model_stats.to_dict('index')
    
    # Stats by hyperparameters
    for param in ['heads', 'noise', 'visual_range']:
        if param in df.columns:
            param_stats = df.groupby(param)['final_loss'].agg(['count', 'mean', 'std', 'min'])
            stats[f'by_{param}'] = param_stats.to_dict('index')
    
    return stats