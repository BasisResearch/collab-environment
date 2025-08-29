#!/usr/bin/env python3
"""
Example usage of GNN explainability analysis
"""

import subprocess

def run_example(description, cmd):
    """Run and display an example command"""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"{'='*60}")
    print(f"Command:\npython {' '.join(cmd)}")
    print("\nRunning...")
    
    result = subprocess.run(
        ["python"] + cmd,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode == 0:
        print("✅ SUCCESS")
    else:
        print("❌ FAILED")
        if result.stderr:
            print(f"Error: {result.stderr[:200]}")

def main():
    """Run example commands"""
    print("GNN Explainability Examples")
    print("=" * 60)
    
    examples = [
        ("Quick Saliency Analysis", [
            "collab_env/gnn/explain_gnn_integrated_gradients.py",
            "--data-name", "boid_single_species_weakalignment_large",
            "--method", "saliency",
            "--max-frames", "3",
            "--output-prefix", "example_saliency"
        ]),
        ("IntegratedGradients Analysis", [
            "collab_env/gnn/explain_gnn_integrated_gradients.py",
            "--data-name", "boid_single_species_weakalignment_large", 
            "--method", "integrated_gradients",
            "--n-steps", "10",
            "--max-frames", "2",
            "--output-prefix", "example_ig"
        ]),
        ("Multi-file Analysis with Caching", [
            "collab_env/gnn/explain_gnn_integrated_gradients.py",
            "--data-name", "boid_single_species_weakalignment_large",
            "--file-id", "-1",
            "--method", "saliency",
            "--max-frames", "2",
            "--output-prefix", "example_multifile"
        ])
    ]
    
    for description, cmd in examples:
        run_example(description, cmd)
    
    print(f"\n{'='*60}")
    print("Examples completed!")
    print("Check generated PNG files for visualizations")

if __name__ == "__main__":
    main()