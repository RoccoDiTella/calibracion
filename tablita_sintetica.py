#!/usr/bin/env python3
"""
Synthetic Data Validation for MMLU Calibration Analysis
Tests different calibration methods on synthetic data with known ground truth
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
            max_iter=100,
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
        patience = 0
        max_patience = 5
        
        for i in range(max_iter // 100):
            loss = optimizer.step(closure)
            current_loss = loss.item()
            
            if verbose and i % 10 == 0:
                print(f"  Iteration {i*100}: loss = {current_loss:.8f}")
            
            if abs(best_loss - current_loss) < tolerance:
                patience += 1
                if patience >= max_patience:
                    if verbose:
                        print(f"  Converged at iteration {i*100}")
                    break
            else:
                patience = 0
                
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
                           calibration_scenario='miscalibrated',
                           noise_level=0.1, seed=42):
    """
    Generate synthetic data simulating a miscalibrated model.
    
    calibration_scenario: 'miscalibrated', 'overconfident', 'underconfident', 'subject_varying'
    """
    np.random.seed(seed)
    
    # Define how the model is miscalibrated
    if calibration_scenario == 'overconfident':
        # Model is overconfident - needs temperature scaling down
        model_temp = 0.5  # Makes predictions too sharp
        model_bias = np.zeros(n_classes)
    elif calibration_scenario == 'underconfident':
        # Model is underconfident - needs temperature scaling up
        model_temp = 2.0  # Makes predictions too flat
        model_bias = np.zeros(n_classes)
    elif calibration_scenario == 'subject_varying':
        # Different subjects have different calibration issues
        model_temp = np.random.uniform(0.5, 2.0, n_subjects)
        model_bias = np.random.uniform(-0.5, 0.5, (n_subjects, n_classes))
        model_bias[:, -1] = 0  # For identifiability
    else:  # 'miscalibrated'
        # General miscalibration with both scale and shift issues
        model_temp = 0.7
        model_bias = np.array([0.3, 0.1, -0.1, 0.0])
    
    X_logits = []
    y_labels = []
    subject_indices = []
    
    for i in range(N):
        subj = np.random.randint(0, n_subjects)
        y = np.random.randint(0, n_classes)
        
        # Generate "true" well-calibrated logits
        true_logits = np.random.randn(n_classes) * 1.0
        true_logits[y] += 2.5  # Signal for correct answer
        
        # Apply miscalibration to get observed logits
        if calibration_scenario == 'subject_varying':
            temp = model_temp[subj]
            bias = model_bias[subj]
        else:
            temp = model_temp
            bias = model_bias if isinstance(model_bias, np.ndarray) else np.zeros(n_classes)
        
        # Model outputs: miscalibrated logits
        observed_logits = true_logits / temp + bias
        
        # Add observation noise
        observed_logits += np.random.randn(n_classes) * noise_level
        
        X_logits.append(observed_logits)
        y_labels.append(y)
        subject_indices.append(subj)
    
    # Determine ground truth calibration parameters
    # These are the parameters that would correct the miscalibration
    if calibration_scenario == 'overconfident':
        true_a = np.ones(n_subjects) * 2.0  # Need to scale up (inverse of 0.5)
        true_b = np.zeros((n_subjects, n_classes))
    elif calibration_scenario == 'underconfident':
        true_a = np.ones(n_subjects) * 0.5  # Need to scale down (inverse of 2.0)
        true_b = np.zeros((n_subjects, n_classes))
    elif calibration_scenario == 'subject_varying':
        true_a = 1.0 / model_temp  # Inverse of temperature
        true_b = -model_bias  # Negative of bias
    else:
        true_a = np.ones(n_subjects) * (1.0 / 0.7)
        true_b = np.tile(-model_bias, (n_subjects, 1))
    
    return (np.array(X_logits, dtype=np.float32), 
            np.array(y_labels, dtype=np.int64), 
            np.array(subject_indices, dtype=np.int64),
            {'true_a': true_a, 'true_b': true_b, 'scenario': calibration_scenario})


def compute_metrics(probs, targets):
    """Compute accuracy and NCE"""
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == targets)
    
    # NCE (Normalized Cross Entropy)
    epsilon = 1e-15
    correct_probs = probs[np.arange(len(targets)), targets]
    nll = -np.mean(np.log(correct_probs + epsilon))
    nce = nll / np.log(4)  # Normalize by uniform entropy
    
    return {'accuracy': acc, 'nce': nce, 'nll': nll}


def analyze_calibration(probs, targets, n_bins=10):
    """Analyze calibration with ECE and reliability diagram data"""
    # Get predicted class and confidence
    pred_class = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)
    correct = (pred_class == targets)
    
    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (pred_conf > bin_lower) & (pred_conf <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = pred_conf[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
            
            bin_data.append({
                'confidence': avg_confidence,
                'accuracy': avg_accuracy,
                'count': in_bin.sum(),
                'bin_center': (bin_lower + bin_upper) / 2
            })
    
    return {'ece': ece, 'bin_data': bin_data}


def run_experiment(train_data, eval_data, config_name, transform, n_subjects, n_seeds=3):
    """Run experiment with multiple seeds and return best result"""
    train_logits, train_targets, train_subjects = train_data
    eval_logits, eval_targets, eval_subjects = eval_data
    
    # Parse config
    share_a = 'a_global' in config_name
    share_b = 'b_global' in config_name
    shift_then_scale = (transform == 'a*(logits+b)')
    
    best_loss = float('inf')
    best_model_state = None
    best_eval_nce = float('inf')
    
    # Try multiple seeds and learning rates
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for lr in [0.1, 0.5, 1.0, 2.0]:
            model = AffineCalibrator(
                n_classes=4, n_subjects=n_subjects,
                share_a=share_a, share_b=share_b,
                shift_then_scale=shift_then_scale
            )
            
            # Add small random initialization to break symmetry
            with torch.no_grad():
                model.log_a.data += torch.randn_like(model.log_a) * 0.1
                model.bias.data += torch.randn_like(model.bias) * 0.1
            
            final_loss = model.fit(
                train_logits, train_targets, train_subjects,
                lr=lr, max_iter=10000, tolerance=1e-16, verbose=False
            )
            
            # Evaluate
            eval_probs = model.predict_proba(eval_logits, eval_subjects)
            eval_metrics = compute_metrics(eval_probs, eval_targets)
            
            if eval_metrics['nce'] < best_eval_nce:
                best_eval_nce = eval_metrics['nce']
                best_loss = final_loss
                best_model_state = model.state_dict()
    
    # Create final model with best parameters
    final_model = AffineCalibrator(
        n_classes=4, n_subjects=n_subjects,
        share_a=share_a, share_b=share_b,
        shift_then_scale=shift_then_scale
    )
    final_model.load_state_dict(best_model_state)
    
    # Final evaluation
    train_probs = final_model.predict_proba(train_logits, train_subjects)
    eval_probs = final_model.predict_proba(eval_logits, eval_subjects)
    
    train_metrics = compute_metrics(train_probs, train_targets)
    eval_metrics = compute_metrics(eval_probs, eval_targets)
    
    # Calibration analysis
    eval_calib = analyze_calibration(eval_probs, eval_targets)
    
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
        'eval_ece': eval_calib['ece'],
        'final_loss': best_loss,
        'n_params': sum(p.numel() for p in final_model.parameters()),
        'a_vals': a_vals,
        'b_vals': b_vals,
        'a_mean': np.mean(a_vals),
        'a_std': np.std(a_vals) if len(a_vals) > 1 else 0,
        'b_mean': np.mean(b_vals)
    }
    
    return result


def plot_calibration_results(results_df, scenario_name):
    """Create visualization of calibration results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. NCE comparison
    ax = axes[0, 0]
    pivot = results_df.pivot_table(
        index='config',
        columns='transform',
        values='eval_nce',
        aggfunc='first'
    )
    pivot.plot(kind='bar', ax=ax)
    ax.set_title('Evaluation NCE by Configuration')
    ax.set_ylabel('NCE (lower is better)')
    ax.legend(title='Transform')
    
    # 2. ECE comparison
    ax = axes[0, 1]
    pivot_ece = results_df.pivot_table(
        index='config',
        columns='transform',
        values='eval_ece',
        aggfunc='first'
    )
    pivot_ece.plot(kind='bar', ax=ax)
    ax.set_title('Expected Calibration Error')
    ax.set_ylabel('ECE (lower is better)')
    ax.legend(title='Transform')
    
    # 3. Parameter count
    ax = axes[1, 0]
    results_df.plot.scatter(x='n_params', y='eval_nce', ax=ax)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Evaluation NCE')
    ax.set_title('Model Complexity vs Performance')
    
    # 4. Best configuration info
    ax = axes[1, 1]
    ax.axis('off')
    best_idx = results_df['eval_nce'].idxmin()
    best = results_df.iloc[best_idx]
    info_text = f"Scenario: {scenario_name}\n\n"
    info_text += f"Best Configuration:\n"
    info_text += f"  {best['config']} - {best['transform']}\n\n"
    info_text += f"Performance:\n"
    info_text += f"  NCE: {best['eval_nce']:.4f}\n"
    info_text += f"  ECE: {best['eval_ece']:.4f}\n"
    info_text += f"  Accuracy: {best['eval_acc']:.4f}\n\n"
    info_text += f"Parameters:\n"
    info_text += f"  Count: {best['n_params']}\n"
    info_text += f"  Mean a: {best['a_mean']:.3f} ± {best['a_std']:.3f}"
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, family='monospace')
    
    plt.tight_layout()
    return fig


def main():
    """Main validation experiment"""
    print("=" * 80)
    print("SYNTHETIC DATA VALIDATION FOR CALIBRATION METHODS")
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
    
    # Test different miscalibration scenarios
    scenarios = ['overconfident', 'underconfident', 'subject_varying', 'miscalibrated']
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*60}")
        
        # Generate data
        N = 5000
        n_subjects = 10
        
        X_logits, y_labels, subject_indices, gt_info = generate_synthetic_data(
            N=N, n_subjects=n_subjects,
            calibration_scenario=scenario,
            noise_level=0.1,
            seed=42
        )
        
        print(f"Generated {N} samples with {n_subjects} subjects")
        print(f"Ground truth scale (a): mean={np.mean(gt_info['true_a']):.3f}, "
              f"std={np.std(gt_info['true_a']):.3f}")
        
        # Split data
        split = int(0.8 * N)
        train_data = (X_logits[:split], y_labels[:split], subject_indices[:split])
        eval_data = (X_logits[split:], y_labels[split:], subject_indices[split:])
        
        # Baseline performance
        eval_probs_uncal = softmax(eval_data[0], axis=1)
        baseline_metrics = compute_metrics(eval_probs_uncal, eval_data[1])
        baseline_calib = analyze_calibration(eval_probs_uncal, eval_data[1])
        
        print(f"\nUncalibrated performance:")
        print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  NCE: {baseline_metrics['nce']:.4f}")
        print(f"  ECE: {baseline_calib['ece']:.4f}")
        
        # Test all configurations
        scenario_results = []
        
        for config_name, transform in configs:
            print(f"\nTesting {config_name} - {transform}...")
            
            result = run_experiment(train_data, eval_data, config_name, transform, n_subjects)
            result['scenario'] = scenario
            result['baseline_nce'] = baseline_metrics['nce']
            result['baseline_ece'] = baseline_calib['ece']
            scenario_results.append(result)
            
            print(f"  NCE: {result['eval_nce']:.4f} (Δ={baseline_metrics['nce']-result['eval_nce']:.4f})")
            print(f"  ECE: {result['eval_ece']:.4f} (Δ={baseline_calib['ece']-result['eval_ece']:.4f})")
        
        # Create visualization
        scenario_df = pd.DataFrame(scenario_results)
        fig = plot_calibration_results(scenario_df, scenario)
        plt.savefig(f'calibration_results_{scenario}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Find best configuration
        best_idx = scenario_df['eval_nce'].idxmin()
        best = scenario_df.iloc[best_idx]
        print(f"\nBest configuration: {best['config']} - {best['transform']}")
        print(f"  NCE improvement: {best['baseline_nce'] - best['eval_nce']:.4f}")
        print(f"  ECE improvement: {best['baseline_ece'] - best['eval_ece']:.4f}")
        
        all_results.extend(scenario_results)
    
    # Summary across all scenarios
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL SCENARIOS")
    print("=" * 80)
    
    all_df = pd.DataFrame(all_results)
    
    # Best configuration for each scenario
    print("\nBest configuration by scenario:")
    for scenario in scenarios:
        scenario_data = all_df[all_df['scenario'] == scenario]
        best_idx = scenario_data['eval_nce'].idxmin()
        best = scenario_data.iloc[best_idx]
        print(f"  {scenario:20s}: {best['config']:20s} {best['transform']:15s} "
              f"(NCE: {best['eval_nce']:.4f})")
    
    # Overall best configuration (most frequent winner)
    best_configs = []
    for scenario in scenarios:
        scenario_data = all_df[all_df['scenario'] == scenario]
        best_idx = scenario_data['eval_nce'].idxmin()
        best = scenario_data.iloc[best_idx]
        best_configs.append((best['config'], best['transform']))
    
    from collections import Counter
    config_counts = Counter(best_configs)
    most_common = config_counts.most_common(1)[0]
    
    print(f"\nMost frequently optimal: {most_common[0][0]} - {most_common[0][1]} "
          f"(won {most_common[1]}/{len(scenarios)} scenarios)")
    
    # Average performance by configuration
    print("\nAverage NCE improvement by configuration:")
    avg_improvement = all_df.groupby(['config', 'transform']).apply(
        lambda x: (x['baseline_nce'] - x['eval_nce']).mean()
    ).sort_values(ascending=False)
    
    for (config, transform), improvement in avg_improvement.items():
        print(f"  {config:20s} {transform:15s}: {improvement:.4f}")
    
    print("\nValidation complete! Check generated plots for detailed results.")


if __name__ == "__main__":
    main()
