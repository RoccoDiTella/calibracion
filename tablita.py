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
                           ground_truth_transform='a*(logits+b)'):
    """Generate synthetic data with known ground truth"""
    
    # Set ground truth parameters
    if ground_truth_config == 'a_global_b_global':
        true_a = np.ones(n_subjects) * 1.2
        true_b = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (n_subjects, 1))
    elif ground_truth_config == 'a_subject_b_global':
        true_a = np.random.uniform(0.8, 1.5, n_subjects)
        true_b = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (n_subjects, 1))
    else:  # a_subject_b_subject
        true_a = np.random.uniform(0.8, 1.5, n_subjects)
        true_b = np.random.uniform(-1, 3, (n_subjects, n_classes))
    
    print(f"Ground truth: {ground_truth_config} - {ground_truth_transform}")
    print(f"True a: mean={np.mean(true_a):.3f}, std={np.std(true_a):.3f}")
    print(f"True b: mean={np.mean(true_b, axis=0)}")
    
    # Generate data
    X_logits = []
    y_labels = []
    subject_indices = []
    
    for i in range(N):
        # Random subject and correct class
        subj = np.random.randint(0, n_subjects)
        y = np.random.randint(0, n_classes)
        
        # Generate signal
        signal = np.random.randn(n_classes) * 2
        signal[y] += np.random.normal(5, 2)  # Add info to correct answer
        
        # Apply inverse calibration to get raw logits
        a = true_a[subj]
        b = true_b[subj]
        
        if ground_truth_transform == 'a*(logits+b)':
            # signal = a * (logits + b)
            # logits = signal/a - b
            logits = signal / a - b
        else:  # a*logits+b
            # signal = a * logits + b
            # logits = (signal - b) / a
            logits = (signal - b) / a
        
        X_logits.append(logits)
        y_labels.append(y)
        subject_indices.append(subj)
    
    return (np.array(X_logits, dtype=np.float32), 
            np.array(y_labels, dtype=np.int64), 
            np.array(subject_indices, dtype=np.int64))


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
    """Run one calibration experiment"""
    train_logits, train_targets, train_subjects = train_data
    eval_logits, eval_targets, eval_subjects = eval_data
    
    # Parse config
    share_a = 'a_global' in config_name
    share_b = 'b_global' in config_name
    shift_then_scale = (transform == 'a*(logits+b)')
    
    # Create and fit model
    model = AffineCalibrator(
        n_classes=4,
        n_subjects=n_subjects,
        share_a=share_a,
        share_b=share_b,
        shift_then_scale=shift_then_scale
    )
    
    final_loss = model.fit(
        train_logits, train_targets, train_subjects,
        lr=1.0, max_iter=5000, tolerance=1e-15, verbose=False
    )
    
    # Evaluate
    train_probs = model.predict_proba(train_logits, train_subjects)
    eval_probs = model.predict_proba(eval_logits, eval_subjects)
    
    train_metrics = compute_metrics(train_probs, train_targets)
    eval_metrics = compute_metrics(eval_probs, eval_targets)
    
    # Get parameters
    with torch.no_grad():
        a_vals = torch.exp(model.log_a).cpu().numpy()
        b_vals = model.bias.cpu().numpy()
    
    result = {
        'config': config_name,
        'transform': transform,
        'train_acc': train_metrics['accuracy'],
        'train_nce': train_metrics['nce'],
        'eval_acc': eval_metrics['accuracy'],
        'eval_nce': eval_metrics['nce'],
        'final_loss': final_loss,
        'n_params': sum(p.numel() for p in model.parameters()),
        'a_mean': np.mean(a_vals),
        'a_std': np.std(a_vals) if len(a_vals) > 1 else 0,
        'b_mean': np.mean(b_vals)
    }
    
    return result


def main():
    """Main experiment"""
    print("=" * 80)
    print("MMLU Calibration Analysis - Clean Implementation")
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
    
    # Test on synthetic data
    print("\nSYNTHETIC DATA EXPERIMENT")
    print("-" * 60)
    
    # Generate data with known ground truth
    N = 5000
    n_subjects = 10
    ground_truth_config = 'a_subject_b_global'
    ground_truth_transform = 'a*(logits+b)'
    
    X_logits, y_labels, subject_indices = generate_synthetic_data(
        N=N, n_subjects=n_subjects,
        ground_truth_config=ground_truth_config,
        ground_truth_transform=ground_truth_transform
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
    print("RESULTS SUMMARY")
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
    
    print("\nTraining NCE (lower is better):")
    print("-" * 60)
    print(pivot['train_nce'].round(4))
    
    print("\nEvaluation Accuracy:")
    print("-" * 60)
    print(pivot['eval_acc'].round(4))
    
    # Show which config matches ground truth
    print(f"\nGround truth was: {ground_truth_config} - {ground_truth_transform}")
    best_config = results_df.loc[results_df['eval_nce'].idxmin()]
    print(f"Best config: {best_config['config']} - {best_config['transform']} (NCE={best_config['eval_nce']:.4f})")
    
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
        # Load MMLU data
        files = glob.glob("mmlu_logits_*.csv")
        if not files:
            print("No MMLU data files found. Skipping MMLU experiment.")
            return
        
        # Group by model and split
        data_by_model = {}
        for file in files:
            df = pd.read_csv(file)
            
            # Extract model and split from filename
            basename = os.path.basename(file)
            parts = basename.replace('mmlu_logits_', '').replace('.csv', '').split('_')
            split = parts[-1]
            model = '_'.join(parts[:-1])
            
            if model not in data_by_model:
                data_by_model[model] = {}
            data_by_model[model][split] = df
        
        # Process each model
        for model_name, splits in data_by_model.items():
            if 'test' not in splits or 'validation' not in splits:
                continue
                
            print(f"\nModel: {model_name}")
            print("-" * 40)
            
            # Prepare data
            def prepare_data(df):
                logits = df[['logit_A', 'logit_B', 'logit_C', 'logit_D']].values.astype(np.float32)
                targets = df['correct_answer_position'].values.astype(np.int64)
                
                # Map subjects to indices
                subjects = sorted(df['subject'].unique())
                subj_to_idx = {s: i for i, s in enumerate(subjects)}
                subj_indices = df['subject'].map(subj_to_idx).values.astype(np.int64)
                
                return logits, targets, subj_indices, len(subjects)
            
            train_logits, train_targets, train_subjects, n_subjects = prepare_data(splits['test'])
            eval_logits, eval_targets, eval_subjects, _ = prepare_data(splits['validation'])
            
            print(f"Train: {len(train_logits)}, Eval: {len(eval_logits)}, Subjects: {n_subjects}")
            
            # Baseline
            eval_probs_uncal = softmax(eval_logits, axis=1)
            baseline = compute_metrics(eval_probs_uncal, eval_targets)
            print(f"Uncalibrated - Accuracy: {baseline['accuracy']:.4f}, NCE: {baseline['nce']:.4f}")
            
            # Run experiments
            mmlu_results = []
            for config_name, transform in configs:
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
            
    except Exception as e:
        print(f"Error processing MMLU data: {e}")
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
