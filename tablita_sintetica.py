import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import itertools

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CalibrationConfig:
    def __init__(self, lr=0.1, max_iter=500, tol=1e-10, device='cpu', 
                 shift_then_scale=True, share_a=False, share_b=True):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.shift_then_scale = shift_then_scale
        self.share_a = share_a
        self.share_b = share_b
        self.history_size = 20
        self.line_search_fn = 'strong_wolfe'

class TorchCalibrator(nn.Module):
    def __init__(self, config, n_topics, n_classes):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        if config.share_a:
            self.loga = nn.Parameter(torch.zeros(1, device=self.device))
        else:
            self.loga = nn.Parameter(torch.zeros(n_topics, device=self.device))
            
        if config.share_b:
            self.b = nn.Parameter(torch.zeros(n_classes, device=self.device))
        else:
            self.b = nn.Parameter(torch.zeros(n_topics, n_classes, device=self.device))

    def forward(self, logits, topics):
        if self.config.share_a:
            scales = torch.exp(self.loga[0]).expand(logits.size(0), 1)
        else:
            scales = torch.exp(self.loga)[topics].unsqueeze(1)
            
        if self.config.share_b:
            b_per = self.b.unsqueeze(0).expand(logits.size(0), -1)
        else:
            b_per = self.b[topics]
            
        if self.config.shift_then_scale:
            return (logits + b_per) * scales
        else:
            return logits * scales + b_per

    def fit(self, logits_np, targets_np, topics_np):
        logits = torch.from_numpy(logits_np).float().to(self.device)
        targets = torch.from_numpy(targets_np).long().to(self.device)
        topics = torch.from_numpy(topics_np).long().to(self.device)
        
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=self.config.lr,
            max_iter=1,
            history_size=self.config.history_size,
            line_search_fn=self.config.line_search_fn
        )
        
        last_loss = float('inf')

        def closure():
            optimizer.zero_grad()
            out = self.forward(logits, topics)
            loss = F.cross_entropy(out, targets)
            loss.backward()
            return loss

        for i in range(1, self.config.max_iter + 1):
            loss = optimizer.step(closure)
            
            with torch.no_grad():
                out = self.forward(logits, topics)
                ce = F.cross_entropy(out, targets).item()
                
            if abs(last_loss - ce) < self.config.tol:
                print(f"  Converged at iteration {i}, loss: {ce:.6f}")
                break
            last_loss = ce
            
        if i == self.config.max_iter:
            print(f"  Max iterations reached, final loss: {ce:.6f}")

    def predict_logits(self, logits_np, topics_np):
        logits = torch.from_numpy(logits_np).float().to(self.device)
        topics = torch.from_numpy(topics_np).long().to(self.device)
        with torch.no_grad():
            out = self.forward(logits, topics)
        return out.cpu().numpy()

def generate_synthetic_data(n_topics=5, n_questions_per_topic=100, n_options=4, train_split=0.7):
    """Generate synthetic MMLU-like data with known biases"""
    
    # True parameters we'll try to recover
    true_bias = np.array([1.0, -0.8, 1.5, -0.7])  # Strong position bias
    topic_difficulties = np.linspace(0.8, 2.5, n_topics)  # Per-topic scaling
    
    all_logits = []
    all_labels = []
    all_topics = []
    all_question_ids = []
    
    question_id = 0
    
    for topic_idx in range(n_topics):
        topic_difficulty = topic_difficulties[topic_idx]
        
        for q_idx in range(n_questions_per_topic):
            # Generate "true" log-likelihoods for this question
            true_logits = np.random.normal(0, 1.0, n_options)
            correct_answer = np.random.randint(0, n_options)
            true_logits[correct_answer] += topic_difficulty  # Topic-dependent boost
            
            # Generate all 24 permutations
            for perm in itertools.permutations(range(n_options)):
                # Apply permutation to true logits
                permuted_true_logits = true_logits[list(perm)]
                
                # Add position bias (this is what we want to recover)
                observed_logits = permuted_true_logits + true_bias
                
                # Find where correct answer ended up after permutation
                new_correct_idx = perm.index(correct_answer)
                
                all_logits.append(observed_logits)
                all_labels.append(new_correct_idx)
                all_topics.append(topic_idx)
                all_question_ids.append(question_id)
            
            question_id += 1
    
    logits = np.array(all_logits)
    labels = np.array(all_labels)
    topics = np.array(all_topics)
    question_ids = np.array(all_question_ids)
    
    # Split by questions (not by permutations) to avoid data leakage
    unique_questions = np.unique(question_ids)
    n_train_questions = int(len(unique_questions) * train_split)
    
    # Shuffle questions for random split
    np.random.shuffle(unique_questions)
    train_questions = unique_questions[:n_train_questions]
    val_questions = unique_questions[n_train_questions:]
    
    # Create train/val masks
    train_mask = np.isin(question_ids, train_questions)
    val_mask = np.isin(question_ids, val_questions)
    
    train_data = {
        'logits': logits[train_mask],
        'labels': labels[train_mask],
        'topics': topics[train_mask],
        'question_ids': question_ids[train_mask]
    }
    
    val_data = {
        'logits': logits[val_mask],
        'labels': labels[val_mask],
        'topics': topics[val_mask],
        'question_ids': question_ids[val_mask]
    }
    
    print(f"Generated {len(logits)} total samples:")
    print(f"  - {n_topics} topics")
    print(f"  - {n_questions_per_topic} questions per topic")
    print(f"  - 24 permutations per question")
    print(f"  - Train: {len(train_data['logits'])} samples ({len(train_questions)} questions)")
    print(f"  - Validation: {len(val_data['logits'])} samples ({len(val_questions)} questions)")
    print(f"  - True bias: {true_bias}")
    print(f"  - Topic difficulties: {topic_difficulties}")
    
    return train_data, val_data, true_bias, topic_difficulties

def evaluate_performance(logits, labels, topics, calibrator):
    """Evaluate calibration performance with Normalized Cross Entropy"""
    calibrated_logits = calibrator.predict_logits(logits, topics)
    
    # Calculate accuracy
    predictions = np.argmax(calibrated_logits, axis=1)
    accuracy = np.mean(predictions == labels)
    
    # Calculate cross-entropy loss
    probs = np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    probs = np.clip(probs, 1e-12, 1.0)
    ce_loss = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    
    # Calculate Normalized Cross Entropy (NCE)
    # NCE = CE / CE_uniform where CE_uniform = log(num_classes)
    n_classes = calibrated_logits.shape[1]
    ce_uniform = np.log(n_classes)
    nce = ce_loss / ce_uniform
    
    return accuracy, ce_loss, nce

def calculate_uncalibrated_performance(logits, labels):
    """Calculate uncalibrated performance metrics"""
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == labels)
    
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    probs = np.clip(probs, 1e-12, 1.0)
    ce_loss = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    
    n_classes = logits.shape[1]
    ce_uniform = np.log(n_classes)
    nce = ce_loss / ce_uniform
    
    return accuracy, ce_loss, nce

def run_experiment():
    """Run the complete calibration experiment"""
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Generate synthetic data with train/val split
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC DATA")
    print("="*60)
    
    train_data, val_data, true_bias, topic_difficulties = generate_synthetic_data(
        n_topics=11, n_questions_per_topic=50, n_options=4, train_split=0.7
    )
    
    # Calculate uncalibrated performance
    train_acc, train_ce, train_nce = calculate_uncalibrated_performance(
        train_data['logits'], train_data['labels']
    )
    val_acc, val_ce, val_nce = calculate_uncalibrated_performance(
        val_data['logits'], val_data['labels']
    )
    
    print(f"\nUncalibrated performance:")
    print(f"  Train - Accuracy: {train_acc:.4f}, CE: {train_ce:.4f}, NCE: {train_nce:.4f}")
    print(f"  Val   - Accuracy: {val_acc:.4f}, CE: {val_ce:.4f}, NCE: {val_nce:.4f}")
    
    # Test configurations
    configs = [
        ('a_shared', 'b_shared', True, True),
        ('a_topic', 'b_shared', False, True),
        ('a_topic', 'b_topic', False, False),
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("RUNNING CALIBRATION EXPERIMENTS")
    print("="*60)
    
    for config_name, b_name, shared_a, shared_b in configs:
        print(f"\nConfiguration: {config_name}, {b_name}")
        print("-" * 40)
        
        # Linear calibration: a * logits + b
        print("Optimizing a * logits + b...")
        linear_config = CalibrationConfig(
            shift_then_scale=False, share_a=shared_a, share_b=shared_b, device=device
        )
        linear_calibrator = TorchCalibrator(linear_config, 11, 4)
        linear_calibrator.fit(train_data['logits'], train_data['labels'], train_data['topics'])
        
        # Evaluate on both train and validation
        linear_train_acc, linear_train_ce, linear_train_nce = evaluate_performance(
            train_data['logits'], train_data['labels'], train_data['topics'], linear_calibrator
        )
        linear_val_acc, linear_val_ce, linear_val_nce = evaluate_performance(
            val_data['logits'], val_data['labels'], val_data['topics'], linear_calibrator
        )
        
        # Shift-scale calibration: a * (logits + b)
        print("Optimizing a * (logits + b)...")
        shift_config = CalibrationConfig(
            shift_then_scale=True, share_a=shared_a, share_b=shared_b, device=device
        )
        shift_calibrator = TorchCalibrator(shift_config, 11, 4)
        shift_calibrator.fit(train_data['logits'], train_data['labels'], train_data['topics'])
        
        # Evaluate on both train and validation
        shift_train_acc, shift_train_ce, shift_train_nce = evaluate_performance(
            train_data['logits'], train_data['labels'], train_data['topics'], shift_calibrator
        )
        shift_val_acc, shift_val_ce, shift_val_nce = evaluate_performance(
            val_data['logits'], val_data['labels'], val_data['topics'], shift_calibrator
        )
        
        results.append({
            'Configuration': f"{config_name}, {b_name}",
            'Uncalibrated_Val': val_nce,
            'Linear_Train': linear_train_nce,
            'Linear_Val': linear_val_nce,
            'Shift_Train': shift_train_nce,
            'Shift_Val': shift_val_nce,
            'Linear_Val_Acc': linear_val_acc,
            'Shift_Val_Acc': shift_val_acc
        })
        
        print(f"  Linear (a*logit+b):")
        print(f"    Train - Acc: {linear_train_acc:.4f}, NCE: {linear_train_nce:.4f}")
        print(f"    Val   - Acc: {linear_val_acc:.4f}, NCE: {linear_val_nce:.4f}")
        print(f"  Shift-Scale (a*(logit+b)):")
        print(f"    Train - Acc: {shift_train_acc:.4f}, NCE: {shift_train_nce:.4f}")
        print(f"    Val   - Acc: {shift_val_acc:.4f}, NCE: {shift_val_nce:.4f}")
    
    # Create results tables for both train and validation
    print("\n" + "="*80)
    print("TRAINING SET RESULTS (Normalized Cross Entropy)")
    print("="*80)
    
    df_train = pd.DataFrame(results)
    df_train_display = df_train[['Configuration', 'Uncalibrated_Val', 'Linear_Train', 'Shift_Train']].copy()
    df_train_display.columns = ['Configuration', 'Uncalibrated', 'a*logit+b', 'a*(logit+b)']
    # Replace uncalibrated val with train for consistency
    df_train_display['Uncalibrated'] = train_nce
    
    print(df_train_display.to_string(index=False, float_format='%.4f'))
    
    print("\n" + "="*80)
    print("VALIDATION SET RESULTS (Normalized Cross Entropy)")
    print("="*80)
    
    df_val = pd.DataFrame(results)
    df_val_display = df_val[['Configuration', 'Uncalibrated_Val', 'Linear_Val', 'Shift_Val']].copy()
    df_val_display.columns = ['Configuration', 'Uncalibrated', 'a*logit+b', 'a*(logit+b)']
    
    print(df_val_display.to_string(index=False, float_format='%.4f'))
    
    # Show parameter recovery for the most interesting case
    print("\n" + "="*60)
    print("PARAMETER RECOVERY ANALYSIS")
    print("="*60)
    
    # Focus on the case where b is shared but a is per-topic (most interesting case)
    linear_config = CalibrationConfig(
        shift_then_scale=False, share_a=False, share_b=True, device=device
    )
    linear_calibrator = TorchCalibrator(linear_config, 11, 4)
    linear_calibrator.fit(train_data['logits'], train_data['labels'], train_data['topics'])
    
    shift_config = CalibrationConfig(
        shift_then_scale=True, share_a=False, share_b=True, device=device
    )
    shift_calibrator = TorchCalibrator(shift_config, 11, 4)
    shift_calibrator.fit(train_data['logits'], train_data['labels'], train_data['topics'])
    
    print(f"True bias vector: {true_bias}")
    print(f"True topic difficulties: {topic_difficulties}")
    print()
    
    # Extract parameters
    with torch.no_grad():
        linear_b = linear_calibrator.b.cpu().numpy()
        linear_a = np.exp(linear_calibrator.loga.cpu().numpy())
        
        shift_b = shift_calibrator.b.cpu().numpy()
        shift_a = np.exp(shift_calibrator.loga.cpu().numpy())
    
    print(f"Linear method b (shared):      {linear_b}")
    print(f"Shift-scale method b (shared): {shift_b}")
    print()
    print(f"Linear method a (per-topic):      {linear_a}")
    print(f"Shift-scale method a (per-topic): {shift_a}")
    
    # Compare bias recovery
    print(f"\nBias recovery analysis:")
    linear_bias_error = np.mean((true_bias - linear_b)**2)
    shift_bias_error = np.mean((true_bias - shift_b)**2)
    
    print(f"Linear method bias MSE:      {linear_bias_error:.4f}")
    print(f"Shift-scale method bias MSE: {shift_bias_error:.4f}")
    
    if shift_bias_error < linear_bias_error:
        print("✓ Shift-scale method better recovers the true bias!")
    else:
        print("✗ Linear method better recovers the true bias")
    
    # Compare scaling recovery
    print(f"\nScaling recovery analysis:")
    linear_scale_error = np.mean((topic_difficulties - linear_a)**2)
    shift_scale_error = np.mean((topic_difficulties - shift_a)**2)
    
    print(f"Linear method scaling MSE:      {linear_scale_error:.4f}")
    print(f"Shift-scale method scaling MSE: {shift_scale_error:.4f}")
    
    print(f"\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print("This experiment tests the hypothesis that when bias (b) is shared")
    print("across topics but scaling (a) is per-topic, the shift-scale method")
    print("a*(logit + b) should better recover the true bias than a*logit + b.")
    print("This is because the shift-scale method first removes the position")
    print("bias before applying topic-specific scaling.")
    print()
    print("Key metric: Normalized Cross Entropy (NCE) = CrossEntropy / log(4)")
    print("Lower NCE indicates better calibration performance.")
    
    return df_val_display, true_bias, topic_difficulties

if __name__ == "__main__":
    results_df, true_bias, topic_difficulties = run_experiment()
