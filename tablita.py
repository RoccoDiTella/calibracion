#!/usr/bin/env python3
"""
Clean MMLU Calibration Analysis
Compares a*logits+b vs a*(logits+b) with different parameter sharing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import pandas as pd
from typing import Dict, List, Tuple
import glob
import os


class AffineCalibrator(nn.Module):
    """Affine calibrator: a*logits+b or a*(logits+b)"""
    
    def __init__(self, n_classes=4, n_subjects=1, share_a=True, share_b=True, 
                 shift_then_scale=False, device='cpu'):
        super().__init__()
        self.n_classes = n_classes
        self.n_subjects = n_subjects
        self.share_a = share_a
        self.share_b = share_b
        self.shift_then_scale = shift_then_scale
        self.device = device
        
        # Initialize scale parameter (in log space for stability)
        if share_a:
            self.log_a = nn.Parameter(torch.zeros(1, device=device))
        else:
            self.log_a = nn.Parameter(torch.zeros(n_subjects, device=device))
            
        # Initialize bias parameter
        if share_b:
            self.bias = nn.Parameter(torch.zeros(n_classes, device=device))
        else:
            self.bias = nn.Parameter(torch.zeros(n_subjects, n_classes, device=device))
    
    def forward(self, logits, subject_indices=None):
        """
        Apply calibration.
        logits: [N, n_classes]
        subject_indices: [N] (only needed if not sharing parameters)
        """
        batch_size = logits.shape[0]
        
        # Get scale parameter
        if self.share_a:
            a = torch.exp(self.log_a)  # [1]
            a = a.expand(batch_size, 1)  # [N, 1]
        else:
            a = torch.exp(self.log_a[subject_indices]).unsqueeze(1)  # [N, 1]
        
        # Get bias parameter
        if self.share_b:
            b = self.bias.unsqueeze(0).expand(batch_size, -1)  # [N, n_classes]
        else:
            b = self.bias[subject_indices]  # [N, n_classes]
        
        # Apply transformation
        if self.shift_then_scale:
            # a * (logits + b)
            return a * (logits + b)
        else:
            # a * logits + b
            return a * logits + b
    
    def fit(self, logits_np, targets_np, subject_indices_np=None, 
            lr=1.0, max_iter=5000, tolerance=1e-15, verbose=False):
        """Fit using L-BFGS on full batch"""
        # Convert to tensors
        logits = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets_np, dtype=torch.long, device=self.device)
        
        if subject_indices_np is not None:
            subject_indices = torch.tensor(subject_indices_np, dtype=torch.long, device=self.device)
        else:
            subject_indices = torch.zeros(len(logits), dtype=torch.long, device=self.device)
        
        # L-BFGS optimizer
        optimizer = torch.optim.LBFGS(
            self.parameters(), 
            lr=lr,
            max_iter=100,  # Internal iterations per step
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            optimizer.zero_grad()
            output = self.forward(logits, subject_indices)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            return loss
        
        # Training loop
        best_loss = float('inf')
        for i in range(max_iter // 100):
            loss = optimizer.step(closure)
            current_loss = loss.item()
            
            if verbose and i % 10 == 0:
                print(f"  Iteration {i*100}: loss = {current_loss:.8f}")
            
            if abs(best_loss - current_loss) < tolerance:
                if verbose:
                    print(f"  Converged at iteration {i*100}")
                break
                
            best_loss = min(best_loss, current_loss)
        
        return best_loss
    
    def predict_proba(self, logits_np, subject_indices_np=None):
        """Get calibrated probabilities"""
        logits = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
        
        if subject_indices_np is not None:
            subject_indices = torch.tensor(subject_indices_np, dtype=torch.long, device=self.device)
        else:
            subject_indices = torch.zeros(len(logits), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            calibrated_logits = self.forward(logits, subject_indices)
            probs = F.softmax(calibrated_logits, dim=1)
        
        return probs.cpu().numpy()


def generate_synthetic_data(N=1000, n_classes=4, n_subjects=10, 
                           ground_truth_config='a_subject_b_global',
                           ground_truth_transform='a*(logits+b)',
                           noise_level=0.1, seed=42):
    """Generate synthetic data with known ground truth"""
    np.random.seed(seed)
    
    # Set ground truth parameters with more variety
    if ground_truth_config == 'a_global_b_global':
        true_a = np.ones(n_subjects) * 1.2
        true_b = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (n_subjects, 1))
    elif ground_truth_config == 'a_subject_b_global':
        # Make sure we have real variation
        true_a = np.random.uniform(0.5, 2.0, n_subjects)  # Wider range
        true_b = np.tile(np.array([2.5, 1.5, 0.5, 0.0]), (n_subjects, 1))
    else:  # a_subject_b_subject
        true_a = np.random.uniform(0.5, 2.0, n_subjects)
        true_b = np.random.uniform(-2, 3, (n_subjects, n_classes))
        # Ensure one class has b=0 for identifiability
        true_b[:, -1] = 0
    
    print(f"Ground truth: {ground_truth_config} - {ground_truth_transform}")
    print(f"True a: mean={np.mean(true_a):.3f}, std={np.std(true_a):.3f}, range=[{np.min(true_a):.3f}, {np.max(true_a):.3f}]")
    
    # Format the array values properly
    b_means = np.mean(true_b, axis=0)
    b_stds = np.std(true_b, axis=0)
    print(f"True b: mean={[f'{x:.3f}' for x in b_means]}, std={np.mean(b_stds):.3f}")
    
    # Generate data with realistic structure
    X_logits = []
    y_labels = []
    subject_indices = []
    
    for i in range(N):
        # Random subject and correct class
        subj = np.random.randint(0, n_subjects)
        y = np.random.randint(0, n_classes)
        
        # Generate more realistic logits - some signal, some noise
        base_logits = np.random.randn(n_classes) * 1.0  # Base noise
        
        # Add signal to correct answer (variable strength)
        signal_strength = np.random.uniform(1.0, 4.0)
        base_logits[y] += signal_strength
        
        # Add some confusing signal to other options
        for j in range(n_classes):
            if j != y and np.random.rand() < 0.3:  # 30% chance of confusing signal
                base_logits[j] += np.random.uniform(0.5, 2.0)
        
        # Apply inverse calibration to get what the model would have output
        a = true_a[subj]
        b = true_b[subj]
        
        if ground_truth_transform == 'a*(logits+b)':
            # calibrated = a * (logits + b)
            # logits = calibrated/a - b
            calibrated_logits = base_logits  # This is what we want after calibration
            raw_logits = calibrated_logits / a - b
        else:  # a*logits+b
            # calibrated = a * logits + b
            # logits = (calibrated - b) / a
            calibrated_logits = base_logits
            raw_logits = (calibrated_logits - b) / a
        
        # Add noise to raw logits (imperfect model)
        raw_logits += np.random.randn(n_classes) * noise_level
        
        X_logits.append(raw_logits)
        y_labels.append(y)
        subject_indices.append(subj)
    
    return (np.array(X_logits, dtype=np.float32), 
            np.array(y_labels, dtype=np.int64), 
            np.array(subject_indices, dtype=np.int64),
            {'true_a': true_a, 'true_b': true_b})


def compute_metrics(probs, targets):
    """Compute accuracy and NCE"""
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == targets)
    
    # NCE
    epsilon = 1e-15
    correct_probs = probs[np.arange(len(targets)), targets]
    nll = -np.mean(np.log(correct_probs + epsilon))
    nce = nll / np.log(4)  # Normalize by uniform entropy
    
    return {'accuracy': acc, 'nce': nce}


def run_experiment(train_data, eval_data, config_name, transform, n_subjects):
    """Enhanced run_experiment with better optimization"""
    train_logits, train_targets, train_subjects = train_data
    eval_logits, eval_targets, eval_subjects = eval_data
    
    # Parse config
    share_a = 'a_global' in config_name
    share_b = 'b_global' in config_name
    shift_then_scale = (transform == 'a*(logits+b)')
    
    # Try multiple learning rates and take the best
    best_loss = float('inf')
    best_model_state = None
    
    for lr in [0.1, 0.5, 1.0, 2.0]:
        model = AffineCalibrator(
            n_classes=4, n_subjects=n_subjects,
            share_a=share_a, share_b=share_b,
            shift_then_scale=shift_then_scale
        )
        
        final_loss = model.fit(
            train_logits, train_targets, train_subjects,
            lr=lr, max_iter=10000, tolerance=1e-16, verbose=False
        )
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_model_state = model.state_dict()
    
    # Create final model with best parameters
    final_model = AffineCalibrator(
        n_classes=4, n_subjects=n_subjects,
        share_a=share_a, share_b=share_b,
        shift_then_scale=shift_then_scale
    )
    final_model.load_state_dict(best_model_state)
    
    # Evaluate
    train_probs = final_model.predict_proba(train_logits, train_subjects)
    eval_probs = final_model.predict_proba(eval_logits, eval_subjects)
    
    train_metrics = compute_metrics(train_probs, train_targets)
    eval_metrics = compute_metrics(eval_probs, eval_targets)
    
    # Get parameters
    with torch.no_grad():
        a_vals = torch.exp(final_model.log_a).cpu().numpy()
        b_vals = final_model.bias.cpu().numpy()
    
    result = {
        'config': config_name,
        'transform': transform,
        'train_acc': train_metrics['accuracy'],
        'train_nce': train_metrics['nce'],
        'eval_acc': eval_metrics['accuracy'],
        'eval_nce': eval_metrics['nce'],
        'final_loss': best_loss,
        'n_params': sum(p.numel() for p in final_model.parameters()),
        'a_mean': np.mean(a_vals),
        'a_std': np.std(a_vals) if len(a_vals) > 1 else 0,
        'b_mean': np.mean(b_vals)
    }
    
    return result


def validate_synthetic_experiment():
    """Comprehensive validation on synthetic data"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SYNTHETIC DATA VALIDATION")
    print("=" * 80)
    
    # Test different ground truth scenarios
    test_scenarios = [
        ('a_global_b_global', 'a*logits+b'),
        ('a_global_b_global', 'a*(logits+b)'),
        ('a_subject_b_global', 'a*logits+b'),
        ('a_subject_b_global', 'a*(logits+b)'),
        ('a_subject_b_subject', 'a*logits+b'),
        ('a_subject_b_subject', 'a*(logits+b)'),
    ]
    
    configs = [
        ('a_global_b_global', 'a*logits+b'),
        ('a_global_b_global', 'a*(logits+b)'),
        ('a_subject_b_global', 'a*logits+b'),
        ('a_subject_b_global', 'a*(logits+b)'),
        ('a_subject_b_subject', 'a*logits+b'),
        ('a_subject_b_subject', 'a*(logits+b)'),
    ]
    
    validation_results = []
    
    for gt_config, gt_transform in test_scenarios:
        print(f"\n{'='*60}")
        print(f"GROUND TRUTH: {gt_config} - {gt_transform}")
        print(f"{'='*60}")
        
        # Generate data
        N = 8000  # Larger dataset
        n_subjects = 15
        
        X_logits, y_labels, subject_indices, gt_params = generate_synthetic_data(
            N=N, n_subjects=n_subjects,
            ground_truth_config=gt_config,
            ground_truth_transform=gt_transform,
            noise_level=0.05,  # Lower noise for cleaner signal
            seed=42
        )
        
        # Split data
        split = int(0.8 * N)
        train_data = (X_logits[:split], y_labels[:split], subject_indices[:split])
        eval_data = (X_logits[split:], y_labels[split:], subject_indices[split:])
        
        # Baseline performance
        eval_probs_uncal = softmax(eval_data[0], axis=1)
        baseline_metrics = compute_metrics(eval_probs_uncal, eval_data[1])
        print(f"Uncalibrated - Accuracy: {baseline_metrics['accuracy']:.4f}, NCE: {baseline_metrics['nce']:.4f}")
        
        # Test all configurations
        scenario_results = []
        best_nce = float('inf')
        best_config = None
        
        for config_name, transform in configs:
            print(f"  Testing {config_name} - {transform}... ", end='')
            
            # Multiple random initializations to ensure we find global optimum
            best_loss = float('inf')
            best_result = None
            
            for seed in range(5):  # Try 5 different initializations
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                result = run_experiment(train_data, eval_data, config_name, transform, n_subjects)
                if result['final_loss'] < best_loss:
                    best_loss = result['final_loss']
                    best_result = result
            
            scenario_results.append(best_result)
            
            # Track overall best
            if best_result['eval_nce'] < best_nce:
                best_nce = best_result['eval_nce']
                best_config = (config_name, transform)
            
            print(f"NCE: {best_result['eval_nce']:.4f}")
        
        # Check if we recovered the correct configuration
        correct_recovery = (best_config[0] == gt_config and best_config[1] == gt_transform)
        
        print(f"\nBest recovered: {best_config[0]} - {best_config[1]} (NCE: {best_nce:.4f})")
        print(f"Correct recovery: {'✓' if correct_recovery else '✗'}")
        
        # Parameter recovery analysis for the correct model
        if correct_recovery:
            correct_result = [r for r in scenario_results 
                            if r['config'] == gt_config and r['transform'] == gt_transform][0]
            
            # Get fitted parameters
            share_a = 'a_global' in gt_config
            share_b = 'b_global' in gt_config
            shift_then_scale = (gt_transform == 'a*(logits+b)')
            
            model = AffineCalibrator(
                n_classes=4, n_subjects=n_subjects,
                share_a=share_a, share_b=share_b,
                shift_then_scale=shift_then_scale
            )
            model.fit(train_data[0], train_data[1], train_data[2], verbose=False)
            
            with torch.no_grad():
                fitted_a = torch.exp(model.log_a).cpu().numpy()
                fitted_b = model.bias.cpu().numpy()
            
            # Compare with ground truth
            true_a = gt_params['true_a']
            true_b = gt_params['true_b']
            
            if share_a:
                a_error = abs(fitted_a[0] - np.mean(true_a))
                print(f"Scale parameter - True: {np.mean(true_a):.3f}, Fitted: {fitted_a[0]:.3f}, Error: {a_error:.3f}")
            else:
                a_errors = np.abs(fitted_a - true_a)
                print(f"Scale parameters - Mean error: {np.mean(a_errors):.3f}, Max error: {np.max(a_errors):.3f}")
            
            if share_b:
                b_errors = np.abs(fitted_b - true_b[0])
                print(f"Bias parameters - Mean error: {np.mean(b_errors):.3f}, Max error: {np.max(b_errors):.3f}")
            else:
                b_errors = np.abs(fitted_b - true_b)
                print(f"Bias parameters - Mean error: {np.mean(b_errors):.3f}, Max error: {np.max(b_errors):.3f}")
        
        # Store results
        validation_results.append({
            'gt_config': gt_config,
            'gt_transform': gt_transform,
            'best_config': best_config[0],
            'best_transform': best_config[1],
            'best_nce': best_nce,
            'correct_recovery': correct_recovery,
            'baseline_nce': baseline_metrics['nce'],
            'improvement': baseline_metrics['nce'] - best_nce
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    validation_df = pd.DataFrame(validation_results)
    
    print("Configuration Recovery Results:")
    print("-" * 50)
    for _, row in validation_df.iterrows():
        status = "✓" if row['correct_recovery'] else "✗"
        print(f"{status} {row['gt_config']:20s} {row['gt_transform']:15s} -> "
              f"{row['best_config']:20s} {row['best_transform']:15s} "
              f"(NCE: {row['best_nce']:.4f}, Δ: {row['improvement']:.4f})")
    
    recovery_rate = validation_df['correct_recovery'].mean()
    avg_improvement = validation_df['improvement'].mean()
    
    print(f"\nOverall Recovery Rate: {recovery_rate:.1%}")
    print(f"Average NCE Improvement: {avg_improvement:.4f}")
    
    if recovery_rate < 0.8:
        print("\n⚠️  WARNING: Low recovery rate suggests optimization issues!")
        print("   Consider: longer training, better initialization, or different optimizer")
    else:
        print("\n✓ Good recovery rate - optimization appears reliable")
    
    return validation_df


def main():
    """Main experiment"""
    print("=" * 80)
    print("MMLU Calibration Analysis - Enhanced with Comprehensive Validation")
    print("=" * 80)
    
    # Configurations to test
    configs = [
        ('a_global_b_global', 'a*logits+b'),
        ('a_global_b_global', 'a*(logits+b)'),
        ('a_subject_b_global', 'a*logits+b'),
        ('a_subject_b_global', 'a*(logits+b)'),
        ('a_subject_b_subject', 'a*logits+b'),
        ('a_subject_b_subject', 'a*(logits+b)'),
    ]
    
    # Comprehensive synthetic validation
    validation_df = validate_synthetic_experiment()
    
    # Quick demonstration with one scenario
    print("\n" + "=" * 60)
    print("QUICK DEMONSTRATION")
    print("=" * 60)
    
    # Generate data with known ground truth
    N = 3000
    n_subjects = 8
    ground_truth_config = 'a_subject_b_global'
    ground_truth_transform = 'a*(logits+b)'
    
    X_logits, y_labels, subject_indices, gt_params = generate_synthetic_data(
        N=N, n_subjects=n_subjects,
        ground_truth_config=ground_truth_config,
        ground_truth_transform=ground_truth_transform,
        noise_level=0.1
    )
    
    # Split data
    split = int(0.8 * N)
    train_data = (X_logits[:split], y_labels[:split], subject_indices[:split])
    eval_data = (X_logits[split:], y_labels[split:], subject_indices[split:])
    
    print(f"\nTrain size: {split}, Eval size: {N-split}")
    
    # Compute baseline
    eval_probs_uncal = softmax(eval_data[0], axis=1)
    baseline_metrics = compute_metrics(eval_probs_uncal, eval_data[1])
    print(f"Uncalibrated - Accuracy: {baseline_metrics['accuracy']:.4f}, NCE: {baseline_metrics['nce']:.4f}")
    
    # Run experiments
    results = []
    print("\nRunning calibration experiments...")
    for config_name, transform in configs:
        print(f"  {config_name} - {transform}")
        result = run_experiment(train_data, eval_data, config_name, transform, n_subjects)
        results.append(result)
    
    # Display results
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("QUICK DEMO RESULTS")
    print("=" * 80)
    
    # Create comparison table
    pivot = results_df.pivot_table(
        index='config',
        columns='transform',
        values=['eval_acc', 'eval_nce', 'train_nce'],
        aggfunc='first'
    )
    
    print("\nEvaluation NCE (lower is better):")
    print("-" * 60)
    print(pivot['eval_nce'].round(4))
    
    print("\nEvaluation Accuracy:")
    print("-" * 60)
    print(pivot['eval_acc'].round(4))
    
    # Show which config matches ground truth
    print(f"\nGround truth was: {ground_truth_config} - {ground_truth_transform}")
    best_config = results_df.loc[results_df['eval_nce'].idxmin()]
    print(f"Best config: {best_config['config']} - {best_config['transform']} (NCE={best_config['eval_nce']:.4f})")
    
    correct_identification = (best_config['config'] == ground_truth_config and 
                            best_config['transform'] == ground_truth_transform)
    print(f"Correctly identified: {'✓' if correct_identification else '✗'}")
    
    # Parameter details
    print("\nParameter counts and convergence:")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['config']:20s} {row['transform']:15s} - "
              f"params: {row['n_params']:3d}, final_loss: {row['final_loss']:.6f}, "
              f"a: {row['a_mean']:.3f}±{row['a_std']:.3f}")
    
    # Try MMLU data if available
    print("\n" + "=" * 80)
    print("MMLU DATA EXPERIMENT")
    print("=" * 80)
    
    try:
        # Load MMLU data - check current directory and common paths
        files = glob.glob("mmlu_logits_*.csv")
        if not files:
            files = glob.glob("*.csv")  # Fallback to any CSV
            files = [f for f in files if "mmlu_logits" in f]
        
        if not files:
            print("No MMLU data files found. Skipping MMLU experiment.")
            print("Expected files like: mmlu_logits_model_name_split.csv")
            return
        
        print(f"Found {len(files)} MMLU files:")
        for f in files:
            print(f"  {f}")
        
        # Group by model and split
        data_by_model = {}
        for file in files:
            try:
                df = pd.read_csv(file)
                print(f"Loading {file}: {len(df)} rows")
                
                # Extract model and split from filename
                basename = os.path.basename(file)
                parts = basename.replace('mmlu_logits_', '').replace('.csv', '').split('_')
                split = parts[-1]
                model = '_'.join(parts[:-1])
                
                # Validate expected columns
                required_cols = ['logit_A', 'logit_B', 'logit_C', 'logit_D', 'correct_answer_position', 'subject']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  Warning: Missing columns in {file}: {missing_cols}")
                    continue
                
                if model not in data_by_model:
                    data_by_model[model] = {}
                data_by_model[model][split] = df
                
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                continue
        
        if not data_by_model:
            print("No valid MMLU data could be loaded.")
            return
        
        # Process each model
        for model_name, splits in data_by_model.items():
            print(f"\nModel: {model_name}")
            print(f"Available splits: {list(splits.keys())}")
            
            # Use test for training, validation for evaluation (MMLU convention)
            train_split = None
            eval_split = None
            
            if 'test' in splits and 'validation' in splits:
                train_split = 'test'
                eval_split = 'validation'
            elif 'test' in splits and 'dev' in splits:
                train_split = 'test'
                eval_split = 'dev'
            elif len(splits) >= 2:
                # Use whatever two splits we have
                split_names = list(splits.keys())
                train_split = split_names[0]
                eval_split = split_names[1]
                print(f"Using {train_split} for training, {eval_split} for evaluation")
            else:
                print(f"  Skipping {model_name}: need at least 2 splits for train/eval")
                continue
            
            print("-" * 40)
            
            # Prepare data
            def prepare_data(df):
                # Check for NaN values in logits
                logit_cols = ['logit_A', 'logit_B', 'logit_C', 'logit_D']
                if df[logit_cols].isna().any().any():
                    print(f"  Warning: Found NaN values in logits, dropping rows")
                    df = df.dropna(subset=logit_cols)
                
                logits = df[logit_cols].values.astype(np.float32)
                targets = df['correct_answer_position'].values.astype(np.int64)
                
                # Map subjects to indices
                subjects = sorted(df['subject'].unique())
                subj_to_idx = {s: i for i, s in enumerate(subjects)}
                subj_indices = df['subject'].map(subj_to_idx).values.astype(np.int64)
                
                return logits, targets, subj_indices, len(subjects)
            
            train_logits, train_targets, train_subjects, n_subjects = prepare_data(splits[train_split])
            eval_logits, eval_targets, eval_subjects, _ = prepare_data(splits[eval_split])
            
            print(f"Train ({train_split}): {len(train_logits)}, Eval ({eval_split}): {len(eval_logits)}, Subjects: {n_subjects}")
            
            # Baseline
            eval_probs_uncal = softmax(eval_logits, axis=1)
            baseline = compute_metrics(eval_probs_uncal, eval_targets)
            print(f"Uncalibrated - Accuracy: {baseline['accuracy']:.4f}, NCE: {baseline['nce']:.4f}")
            
            # Run experiments
            mmlu_results = []
            for config_name, transform in configs:
                print(f"  Running {config_name} - {transform}")
                result = run_experiment(
                    (train_logits, train_targets, train_subjects),
                    (eval_logits, eval_targets, eval_subjects),
                    config_name, transform, n_subjects
                )
                result['model'] = model_name
                mmlu_results.append(result)
            
            # Show results for this model
            mmlu_df = pd.DataFrame(mmlu_results)
            pivot = mmlu_df.pivot_table(
                index='config',
                columns='transform',
                values='eval_nce',
                aggfunc='first'
            )
            print("\nCalibrated NCE:")
            print(pivot.round(4))
            
            # Show best result
            best_idx = mmlu_df['eval_nce'].idxmin()
            best = mmlu_df.iloc[best_idx]
            print(f"Best: {best['config']} - {best['transform']} (NCE: {best['eval_nce']:.4f})")
            
    except Exception as e:
        print(f"Error processing MMLU data: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
