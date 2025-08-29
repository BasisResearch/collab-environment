#!/usr/bin/env python3
"""
Test script for GNN explainability analysis
"""

import sys
import subprocess
import time
import os

def run_test(description, cmd):
    """Run a test command and return success status"""
    print(f"\n--- {description} ---")
    print(f"Command: python {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            ["python"] + cmd,
            capture_output=True,
            text=True,
            timeout=90
        )
        
        if result.returncode == 0:
            print("✓ SUCCESS")
            return True
        else:
            print("✗ FAILED")
            if result.stderr:
                print(f"  Error: {result.stderr[-300:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """Run all tests"""
    print("GNN Explainability Test Suite")
    print("=" * 60)
    
    # Test configurations
    tests = [
        ("IntegratedGradients", [
            "collab_env/gnn/explain_gnn_integrated_gradients.py",
            "--data-name", "boid_single_species_weakalignment_large",
            "--method", "integrated_gradients",
            "--max-frames", "2",
            "--n-steps", "5",
            "--output-prefix", "test_ig",
            "--no-save-data"
        ]),
        ("Saliency", [
            "collab_env/gnn/explain_gnn_integrated_gradients.py", 
            "--data-name", "boid_single_species_weakalignment_large",
            "--method", "saliency",
            "--max-frames", "2",
            "--output-prefix", "test_saliency",
            "--no-save-data"
        ]),
        ("Food Model", [
            "collab_env/gnn/explain_gnn_integrated_gradients.py",
            "--data-name", "boid_food_basic", 
            "--method", "saliency",
            "--max-frames", "1",
            "--output-prefix", "test_food",
            "--no-save-data"
        ]),
    ]
    
    # Clean up test files
    test_files = ["test_ig.png", "test_saliency.png", "test_food.png"]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Run tests
    start_time = time.time()
    results = []
    
    for description, cmd in tests:
        success = run_test(description, cmd)
        results.append((description, success))
    
    # Clean up
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(success for _, success in results)
    for description, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {description}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)