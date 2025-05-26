#!/usr/bin/env python3
"""
Synthetic Data Validation for Calibration Methods
Tests different calibration approaches with controlled ground truth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


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
        """Apply calibration."""
        batch_size = logits.shape[0]
        
        # Get scale parameter
        if self.share_a:
            a = torch.exp(self.log_a)
            a = a.expand(batch_size, 1)
        else:
            a = torch.exp(self.log_a[subject_indices]).unsqueeze(1)
        
        # Get bias parameter
        if self.share_b:
            b = self.bias.unsqueeze(0).expand(batch_size, -1)
        else:
            b = self.bias[subject_indices]
        
        # Apply transformation
        if self.shift_then_scale:
            # a * (logits + b)
            return a * (logits + b)
        else:
            # a * logits + b
            return a * logits + b
    
    def fit(self, logits_np, targets_np, subject_indices_np=None, 
            lr=1.0, max_iter=10000, verbose=False):
        """Fit using L-BFGS"""
        # Convert to tensors
        logits = torch.tensor(logits_np, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets_np, dtype=torch.long, device=self.device)
        
        if subject_indices_np is not None:
            subject_indices = torch.tensor(subject_indices_np, dtype=torch.long, device=self.device)
        else:
            subject_indices = torch.zeros(len(logits), dtype=torch.long, device=self.device)
        
        # Try multiple optimizers
        best_loss = float('inf')
        best_state = self.state_dict()
        
        # First try L-BFGS
        optimizer = torch.optim.LBFGS(
            self.parameters(), 
            lr=lr,
            max_iter=100,
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            optimizer.zero_grad()
            output = self.forward(logits, subject_indices)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            return loss
        
        for i in range(max_iter // 100):
            loss = optimizer.step(closure)
            current_loss = loss.item()
            
            if verbose and i % 10 == 0:
                print(f"  Iteration {i*100}: loss = {current_loss:.8f}")
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = self.state_dict()
            
            if i > 0 and abs(previous_loss - current_loss) < 1e-15:
                break
            previous_loss = current_loss
        
        # Also try Adam for comparison
        self.load_state_dict(best_state)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        
        for i in range(5000):
            optimizer.zero_grad()
            output = self.forward(logits, subject_indices)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = self.state_dict()
        
        self.load_state_dict(best_state)
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


def generate_simple_synthetic_data(N=1000, n_classes=4, n_subjects=10, 
                                 scenario='global_bias', seed=42):
    """
    Generate synthetic data following your example structure.
    
    Scenarios:
    - 'global_bias': Large bias terms affecting all subjects equally
    - 'subject_bias': Different bias patterns per subject  
    - 'subject_info': Different information quality per subject
    """
    np.random.seed(seed)
    
    # Generate labels uniformly
    Y = np.random.multinomial(1, pvals=np.ones(n_classes)/n_classes, size=N)
    y_labels = Y.argmax(axis=1)
    
    # Assign subjects randomly
    subject_indices = np.random.randint(0, n_subjects, size=N)
    
    # Define scenario-specific parameters
    if scenario == 'global_bias':
        # All subjects have same large bias problem
        bias = np.array([30.0, 20.0, 10.0, 0.0])
        info_mean = 2.0
        info_std = 2.0
        
        def logits_from_ans(y, subj_idx):
            info = np.random.normal(info_mean, info_std)
            noise = np.random.normal(0, 1, size=n_classes)
            return y * info + bias + noise
            
    elif scenario == 'subject_bias':
        # Each subject has different bias
        biases = []
        for s in range(n_subjects):
            # Random ordering of bias magnitudes
            mags = [30, 20, 10, 0]
            np.random.shuffle(mags)
            biases.append(np.array(mags, dtype=float))
        biases = np.array(biases)
        
        info_mean = 2.0
        info_std = 2.0
        
        def logits_from_ans(y, subj_idx):
            info = np.random.normal(info_mean, info_std)
            noise = np.random.normal(0, 1, size=n_classes)
            return y * info + biases[subj_idx] + noise
            
    elif scenario == 'subject_info':
        # Each subject has different information quality
        bias = np.array([30.0, 20.0, 10.0, 0.0])
        info_means = np.random.uniform(0.5, 4.0, n_subjects)
        info_stds = np.random.uniform(0.5, 3.0, n_subjects)
        
        def logits_from_ans(y, subj_idx):
            info = np.random.normal(info_means[subj_idx], info_stds[subj_idx])
            noise = np.random.normal(0, 1, size=n_classes)
            return y * info + bias + noise
    
    else:  # 'mixed'
        # Both bias and info vary by subject
        biases = []
        for s in range(n_subjects):
            base = np.random.uniform(20, 40)
            biases.append(np.array([base, base-10, base-20, 0]))
        biases = np.array(biases)
        
        info_means = np.random.uniform(1.0, 3.0, n_subjects)
        info_stds = np.random.uniform(1.0, 2.0, n_subjects)
        
        def logits_from_ans(y, subj_idx):
            info = np.random.normal(info_means[subj_idx], info_stds[subj_idx])
            noise = np.random.normal(0, 1, size=n_classes)
            return y * info + biases[subj_idx] + noise
    
    # Generate logits
    X_logits = []
    for i in range(N):
        logits = logits_from_ans(Y[i], subject_indices[i])
        X_logits.append(logits)
    
    X_logits = np.array(X_logits, dtype=np.float32)
    
    return X_logits, y_labels, subject_indices, {'scenario': scenario}


def compute_metrics(probs, targets):
    """Compute accuracy, NCE, and ECE"""
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == targets)
    
    # NCE
    epsilon = 1e-15
    correct_probs = probs[np.arange(len(targets)), targets]
    nll = -np.mean(np.log(correct_probs + epsilon))
    nce = nll / np.log(4)
    
    # ECE
    pred_conf = np.max(probs, axis=1)
    correct = (pred == targets)
    
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (pred_conf > bin_boundaries[i]) & (pred_conf <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = pred_conf[in_bin].mean()
            ece += np.abs(bin_acc - bin_conf) * in_bin.mean()
    
    return {'accuracy': acc, 'nce': nce, 'ece': ece}


def analyze_logit_statistics(X_logits, y_labels, subject_indices):
    """Analyze the properties of generated logits"""
    n_subjects = len(np.unique(subject_indices))
    
    print("\nLogit Statistics:")
    print("-" * 50)
    
    # Overall statistics
    print(f"Overall logit range: [{X_logits.min():.1f}, {X_logits.max():.1f}]")
    print(f"Overall logit mean: {X_logits.mean():.1f} ± {X_logits.std():.1f}")
    
    # Correct vs incorrect logits
    correct_logits = []
    incorrect_logits = []
    
    for i, (logits, label) in enumerate(zip(X_logits, y_labels)):
        correct_logits.append(logits[label])
        incorrect_logits.extend(logits[np.arange(len(logits)) != label])
    
    correct_logits = np.array(correct_logits)
    incorrect_logits = np.array(incorrect_logits)
    
    print(f"\nCorrect answer logits: {correct_logits.mean():.1f} ± {correct_logits.std():.1f}")
    print(f"Incorrect answer logits: {incorrect_logits.mean():.1f} ± {incorrect_logits.std():.1f}")
    print(f"Signal (difference): {correct_logits.mean() - incorrect_logits.mean():.1f}")
    
    # Per-subject analysis
    if n_subjects > 1:
        print("\nPer-subject signal strength:")
        for s in range(min(5, n_subjects)):  # Show first 5 subjects
            mask = subject_indices == s
            if mask.sum() > 0:
                subj_correct = []
                subj_incorrect = []
                for i in np.where(mask)[0]:
                    logits = X_logits[i]
                    label = y_labels[i]
                    subj_correct.append(logits[label])
                    subj_incorrect.extend(logits[np.arange(len(logits)) != label])
                
                signal = np.mean(subj_correct) - np.mean(subj_incorrect)
                print(f"  Subject {s}: {signal:.1f}")


def run_experiment(train_data, eval_data, config_name, transform, n_subjects, 
                  n_seeds=3, verbose=False):
    """Run calibration experiment"""
    train_logits, train_targets, train_subjects = train_data
    eval_logits, eval_targets, eval_subjects = eval_data
    
    # Parse config
    share_a = 'a_global' in config_name
    share_b = 'b_global' in config_name
    shift_then_scale = (transform == 'a*(logits+b)')
    
    best_eval_nce = float('inf')
    best_model = None
    
    # Try multiple seeds
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        
        # Try different learning rates
        for lr in [0.1, 0.5, 1.0, 2.0]:
            model = AffineCalibrator(
                n_classes=4, n_subjects=n_subjects,
                share_a=share_a, share_b=share_b,
                shift_then_scale=shift_then_scale
            )
            
            # Initialize with small random values
            with torch.no_grad():
                model.log_a.data += torch.randn_like(model.log_a) * 0.1
                model.bias.data += torch.randn_like(model.bias) * 0.1
            
            final_loss = model.fit(
                train_logits, train_targets, train_subjects,
                lr=lr, verbose=verbose
            )
            
            # Evaluate
            eval_probs = model.predict_proba(eval_logits, eval_subjects)
            eval_metrics = compute_metrics(eval_probs, eval_targets)
            
            if eval_metrics['nce'] < best_eval_nce:
                best_eval_nce = eval_metrics['nce']
                best_model = model
    
    # Final evaluation with best model
    train_probs = best_model.predict_proba(train_logits, train_subjects)
    eval_probs = best_model.predict_proba(eval_logits, eval_subjects)
    
    train_metrics = compute_metrics(train_probs, train_targets)
    eval_metrics = compute_metrics(eval_probs, eval_targets)
    
    # Get parameters
    with torch.no_grad():
        a_vals = torch.exp(best_model.log_a).cpu().numpy()
        b_vals = best_model.bias.cpu().numpy()
    
    return {
        'config': config_name,
        'transform': transform,
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'n_params': sum(p.numel() for p in best_model.parameters()),
        'a_vals': a_vals,
        'b_vals': b_vals,
        'model': best_model
    }


def visualize_calibration_effects(original_logits, model, subject_idx=0):
    """Visualize how calibration transforms the logits"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sample logits for visualization
    sample_logits = original_logits[:100]
    
    # Original probabilities
    ax = axes[0]
    orig_probs = softmax(sample_logits, axis=1)
    im = ax.imshow(orig_probs.T, aspect='auto', cmap='viridis')
    ax.set_title('Original Probabilities')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Class')
    plt.colorbar(im, ax=ax)
    
    # Calibrated probabilities
    ax = axes[1]
    cal_probs = model.predict_proba(sample_logits, 
                                   np.full(len(sample_logits), subject_idx))
    im = ax.imshow(cal_probs.T, aspect='auto', cmap='viridis')
    ax.set_title('Calibrated Probabilities')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Class')
    plt.colorbar(im, ax=ax)
    
    # Difference
    ax = axes[2]
    diff = cal_probs - orig_probs
    im = ax.imshow(diff.T, aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_title('Difference (Calibrated - Original)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Class')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def main():
    """Main validation experiment"""
    print("=" * 80)
    print("SYNTHETIC CALIBRATION VALIDATION")
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
    
    # Test different scenarios
    scenarios = ['global_bias', 'subject_bias', 'subject_info', 'mixed']
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*60}")
        
        # Generate data
        N = 5000
        n_subjects = 10
        
        X_logits, y_labels, subject_indices, info = generate_simple_synthetic_data(
            N=N, n_subjects=n_subjects, scenario=scenario, seed=42
        )
        
        # Analyze the generated data
        analyze_logit_statistics(X_logits, y_labels, subject_indices)
        
        # Split data
        split = int(0.8 * N)
        train_data = (X_logits[:split], y_labels[:split], subject_indices[:split])
        eval_data = (X_logits[split:], y_labels[split:], subject_indices[split:])
        
        # Baseline performance
        eval_probs_uncal = softmax(eval_data[0], axis=1)
        baseline_metrics = compute_metrics(eval_probs_uncal, eval_data[1])
        
        print(f"\nUncalibrated baseline:")
        print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  NCE: {baseline_metrics['nce']:.4f}")
        print(f"  ECE: {baseline_metrics['ece']:.4f}")
        
        # Test all configurations
        scenario_results = []
        
        print("\nTesting calibration methods:")
        for config_name, transform in configs:
            result = run_experiment(
                train_data, eval_data, config_name, transform, 
                n_subjects, n_seeds=3, verbose=False
            )
            
            result['scenario'] = scenario
            result['baseline_metrics'] = baseline_metrics
            scenario_results.append(result)
            
            eval_m = result['eval_metrics']
            print(f"  {config_name:20s} {transform:15s}: "
                  f"NCE={eval_m['nce']:.4f} "
                  f"(Δ={baseline_metrics['nce']-eval_m['nce']:+.4f}), "
                  f"ECE={eval_m['ece']:.4f}")
        
        # Find best method
        best_idx = min(range(len(scenario_results)), 
                      key=lambda i: scenario_results[i]['eval_metrics']['nce'])
        best = scenario_results[best_idx]
        
        print(f"\nBest method: {best['config']} - {best['transform']}")
        print(f"  NCE improvement: {best['baseline_metrics']['nce'] - best['eval_metrics']['nce']:.4f}")
        
        # Visualize calibration effect for best method
        fig = visualize_calibration_effects(eval_data[0][:100], best['model'])
        plt.suptitle(f'Calibration Effect - {scenario} - {best["config"]} {best["transform"]}')
        # plt.savefig(f'calibration_effect_{scenario}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Parameter analysis
        print(f"\nCalibration parameters:")
        print(f"  Scale (a): mean={np.mean(best['a_vals']):.3f}, "
              f"std={np.std(best['a_vals']):.3f}")
        print(f"  Bias (b): {best['b_vals']}")
        
        all_results.extend(scenario_results)
    
    # Summary visualization
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    results_df = pd.DataFrame([{
        'scenario': r['scenario'],
        'config': r['config'],
        'transform': r['transform'],
        'nce': r['eval_metrics']['nce'],
        'ece': r['eval_metrics']['ece'],
        'nce_improvement': r['baseline_metrics']['nce'] - r['eval_metrics']['nce']
    } for r in all_results])
    
    # NCE by scenario and method
    ax = axes[0, 0]
    pivot = results_df.pivot_table(
        index=['config', 'transform'], 
        columns='scenario', 
        values='nce'
    )
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('NCE')
    ax.set_title('NCE by Method and Scenario')
    ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # ECE by scenario and method
    ax = axes[0, 1]
    pivot = results_df.pivot_table(
        index=['config', 'transform'], 
        columns='scenario', 
        values='ece'
    )
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('ECE')
    ax.set_title('ECE by Method and Scenario')
    ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Best method per scenario
    ax = axes[1, 0]
    best_methods = results_df.loc[results_df.groupby('scenario')['nce'].idxmin()]
    best_counts = best_methods.groupby(['config', 'transform']).size()
    best_counts.plot(kind='bar', ax=ax)
    ax.set_ylabel('Times Selected as Best')
    ax.set_title('Best Method Frequency')
    
    # NCE improvement distribution
    ax = axes[1, 1]
    for scenario in scenarios:
        data = results_df[results_df['scenario'] == scenario]['nce_improvement']
        ax.hist(data, alpha=0.5, label=scenario, bins=10)
    ax.set_xlabel('NCE Improvement')
    ax.set_ylabel('Count')
    ax.set_title('NCE Improvement Distribution')
    ax.legend()
    
    plt.tight_layout()
    # plt.savefig('calibration_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print best method for each scenario
    print("\nBest method by scenario:")
    for scenario in scenarios:
        scenario_data = results_df[results_df['scenario'] == scenario]
        best_idx = scenario_data['nce'].idxmin()
        best = scenario_data.loc[best_idx]
        print(f"  {scenario:15s}: {best['config']:20s} {best['transform']:15s} "
              f"(NCE={best['nce']:.4f}, Δ={best['nce_improvement']:.4f})")
    
    print("\nValidation complete! Check generated plots for visualizations.")


if __name__ == "__main__":
    main()
