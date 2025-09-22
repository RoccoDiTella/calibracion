import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import math
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1) Config: add dtype, use saner tolerances, larger history
class CalibrationConfig:
    def __init__(self, lr=1.0, max_iter=50, tol=1e-7, device='cpu',
                 shift_then_scale=True, share_a=False, share_b=True,
                 dtype=torch.float64):
        self.lr = lr
        self.max_iter = max_iter         # internal LBFGS iterations
        self.tol = tol                   # used for LBFGS tolerances
        self.device = device
        self.shift_then_scale = shift_then_scale
        self.share_a = share_a
        self.share_b = share_b
        self.history_size = 100
        self.line_search_fn = 'strong_wolfe'
        self.dtype = dtype

# --- 2) TorchCalibrator: carry dtype through parameters
class TorchCalibrator(nn.Module):
    def __init__(self, config, n_topics, n_classes):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        if config.share_a:
            self.loga = nn.Parameter(torch.zeros(1, device=self.device, dtype=self.dtype))
        else:
            self.loga = nn.Parameter(torch.zeros(n_topics, device=self.device, dtype=self.dtype))

        if config.share_b:
            self.b = nn.Parameter(torch.zeros(n_classes, device=self.device, dtype=self.dtype))
        else:
            self.b = nn.Parameter(torch.zeros(n_topics, n_classes, device=self.device, dtype=self.dtype))

    def forward(self, logits, topics):
        if self.config.share_a:
            scales = torch.exp(self.loga[0]).expand(logits.size(0), 1)
        else:
            scales = torch.exp(self.loga)[topics].unsqueeze(1)

        if self.config.share_b:
            b_per = self.b.unsqueeze(0).expand(logits.size(0), -1)
        else:
            b_per = self.b[topics]

        return (logits + b_per) * scales if self.config.shift_then_scale else logits * scales + b_per

    def fit(self, logits_np, targets_np, topics_np, verbose=False):
        # full-batch tensors in float64 for stability
        logits = torch.from_numpy(logits_np).to(self.device, dtype=self.dtype)
        targets = torch.from_numpy(targets_np).to(self.device, dtype=torch.long)
        topics  = torch.from_numpy(topics_np).to(self.device, dtype=torch.long)

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=self.config.lr,
            max_iter=self.config.max_iter,          # let LBFGS iterate internally
            history_size=self.config.history_size,
            line_search_fn=self.config.line_search_fn,
            tolerance_grad=self.config.tol,
            tolerance_change=self.config.tol
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            out = self.forward(logits, topics)
            loss = F.cross_entropy(out, targets)   # no regularization
            loss.backward()
            return loss

        final_loss = optimizer.step(closure).item()
        if verbose:
            print(f"  LBFGS finished, loss: {final_loss:.6f}")

    def predict_logits(self, logits_np, topics_np):
        logits = torch.from_numpy(logits_np).to(self.device, dtype=self.dtype)
        topics = torch.from_numpy(topics_np).to(self.device, dtype=torch.long)
        with torch.no_grad():
            out = self.forward(logits, topics)
        return out.cpu().numpy()

def load_mmlu_data(model_name="meta_llama_Llama_3.2_3B_Instruct"):
    """Load MMLU data from saved CSV files"""
    
    # Try to load test and validation data
    try:
        test_file = f"mmlu_logits_{model_name}_test.csv"
        validation_file = f"mmlu_logits_{model_name}_validation.csv"
        
        test_df = pd.read_csv(test_file)
        validation_df = pd.read_csv(validation_file)
        
        print(f"Loaded MMLU data:")
        print(f"  Test (for training/CV): {len(test_df)} rows")
        print(f"  Validation (for final test): {len(validation_df)} rows")
        
        return test_df, validation_df
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure you have run forward.py with Llama 3.2 3B Instruct first")
        return None, None

def prepare_data(df):
    """Prepare data for calibration"""
    # Extract logits
    logit_cols = ['logit_A', 'logit_B', 'logit_C', 'logit_D']
    logits = df[logit_cols].values
    
    # Extract labels
    labels = df['correct_answer_position'].values
    
    # Encode subjects as topic indices
    label_encoder = LabelEncoder()
    topics = label_encoder.fit_transform(df['subject'].values)
    
    # Get unique subjects for reporting
    unique_subjects = df['subject'].nunique()
    
    print(f"Data preparation:")
    print(f"  Samples: {len(logits)}")
    print(f"  Unique subjects: {unique_subjects}")
    print(f"  Logits shape: {logits.shape}")
    
    return logits, labels, topics, unique_subjects, label_encoder

def evaluate_performance(logits, labels, topics, calibrator):
    """Evaluate calibration performance with NCE"""
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

def run_calibration_cv(logits, labels, topics, n_topics, device='cuda', n_folds=5):
    """Run cross-validation calibration experiment"""
    
    # Test configurations
    configs = [
        ('a_shared', 'b_shared', True, True),
        ('a_topic', 'b_shared', False, True),
        ('a_topic', 'b_topic', False, False),
    ]
    
    # Get unique question IDs to split by questions, not permutations
    # Assume every 24 rows is one question (24 permutations)
    n_samples = len(logits)
    n_questions = n_samples // 24
    question_ids = np.repeat(np.arange(n_questions), 24)
    
    unique_questions = np.unique(question_ids)
    
    cv_results = []
    
    # Cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for config_name, b_name, shared_a, shared_b in configs:
        print(f"Cross-validation for {config_name}, {b_name}...")
        
        linear_nces = []
        shift_nces = []
        
        for fold, (train_q_idx, val_q_idx) in enumerate(kfold.split(unique_questions)):
            train_questions = unique_questions[train_q_idx]
            val_questions = unique_questions[val_q_idx]
            
            # Create masks
            train_mask = np.isin(question_ids, train_questions)
            val_mask = np.isin(question_ids, val_questions)
            
            train_logits = logits[train_mask]
            train_labels = labels[train_mask]
            train_topics = topics[train_mask]
            
            val_logits = logits[val_mask]
            val_labels = labels[val_mask]
            val_topics = topics[val_mask]
            
            # Linear calibration
            linear_config = CalibrationConfig(
                shift_then_scale=False, share_a=shared_a, share_b=shared_b, device=device
            )
            linear_calibrator = TorchCalibrator(linear_config, n_topics, 4)
            linear_calibrator.fit(train_logits, train_labels, train_topics)
            _, _, linear_nce = evaluate_performance(val_logits, val_labels, val_topics, linear_calibrator)
            linear_nces.append(linear_nce)
            
            # Shift-scale calibration
            shift_config = CalibrationConfig(
                shift_then_scale=True, share_a=shared_a, share_b=shared_b, device=device
            )
            shift_calibrator = TorchCalibrator(shift_config, n_topics, 4)
            shift_calibrator.fit(train_logits, train_labels, train_topics)
            _, _, shift_nce = evaluate_performance(val_logits, val_labels, val_topics, shift_calibrator)
            shift_nces.append(shift_nce)
        
        # Calculate means and standard errors
        linear_mean = np.mean(linear_nces)
        linear_se = np.std(linear_nces) / np.sqrt(len(linear_nces))
        
        shift_mean = np.mean(shift_nces)
        shift_se = np.std(shift_nces) / np.sqrt(len(shift_nces))
        
        cv_results.append({
            'Configuration': f"{config_name}, {b_name}",
            'Linear_Mean': linear_mean,
            'Linear_SE': linear_se,
            'Shift_Mean': shift_mean,
            'Shift_SE': shift_se
        })
    
    return cv_results

def run_mmlu_experiment():
    """Run the complete MMLU calibration experiment"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load MMLU data
    print("\n" + "="*60)
    print("LOADING MMLU DATA")
    print("="*60)
    
    test_df, validation_df = load_mmlu_data()
    if test_df is None or validation_df is None:
        return
    
    # Prepare data
    # Use "test" set for training/CV (confusing MMLU naming)
    train_logits, train_labels, train_topics, n_topics, label_encoder = prepare_data(test_df)
    # Use "validation" set for final testing
    test_logits, test_labels, test_topics, _, _ = prepare_data(validation_df)
    
    # Ensure test topics use same encoding
    test_topics_encoded = label_encoder.transform(validation_df['subject'].values)
    
    # Calculate uncalibrated performance
    train_uncal_acc, train_uncal_ce, train_uncal_nce = calculate_uncalibrated_performance(train_logits, train_labels)
    test_uncal_acc, test_uncal_ce, test_uncal_nce = calculate_uncalibrated_performance(test_logits, test_labels)
    
    print(f"\nUncalibrated performance:")
    print(f"  Train: Acc={train_uncal_acc:.4f}, NCE={train_uncal_nce:.4f}")
    print(f"  Test:  Acc={test_uncal_acc:.4f}, NCE={test_uncal_nce:.4f}")
    
    # Test configurations
    configs = [
        ('a_shared', 'b_shared', True, True),
        ('a_topic', 'b_shared', False, True),
        ('a_topic', 'b_topic', False, False),
    ]
    
    print("\n" + "="*60)
    print("RUNNING CALIBRATION EXPERIMENTS")
    print("="*60)
    
    # 1. Full training set results
    print("\nTraining on full training set...")
    full_train_results = []
    
    for config_name, b_name, shared_a, shared_b in configs:
        # Linear calibration
        linear_config = CalibrationConfig(
            shift_then_scale=False, share_a=shared_a, share_b=shared_b, device=device
        )
        linear_calibrator = TorchCalibrator(linear_config, n_topics, 4)
        linear_calibrator.fit(train_logits, train_labels, train_topics, verbose=True)
        _, _, linear_nce = evaluate_performance(train_logits, train_labels, train_topics, linear_calibrator)
        
        # Shift-scale calibration
        shift_config = CalibrationConfig(
            shift_then_scale=True, share_a=shared_a, share_b=shared_b, device=device
        )
        shift_calibrator = TorchCalibrator(shift_config, n_topics, 4)
        shift_calibrator.fit(train_logits, train_labels, train_topics, verbose=True)
        _, _, shift_nce = evaluate_performance(train_logits, train_labels, train_topics, shift_calibrator)
        
        full_train_results.append({
            'Configuration': f"{config_name}, {b_name}",
            'Uncalibrated': train_uncal_nce,
            'a*logit+b': linear_nce,
            'a*(logit+b)': shift_nce
        })
    
    # 2. Cross-validation results
    print("\nRunning cross-validation...")
    cv_results = run_calibration_cv(train_logits, train_labels, train_topics, n_topics, device)
    
    # 3. Final test set results
    print("\nEvaluating on test set...")
    test_results = []
    
    for config_name, b_name, shared_a, shared_b in configs:
        # Train on full training set, evaluate on test
        linear_config = CalibrationConfig(
            shift_then_scale=False, share_a=shared_a, share_b=shared_b, device=device
        )
        linear_calibrator = TorchCalibrator(linear_config, n_topics, 4)
        linear_calibrator.fit(train_logits, train_labels, train_topics)
        _, _, linear_test_nce = evaluate_performance(test_logits, test_labels, test_topics_encoded, linear_calibrator)
        
        shift_config = CalibrationConfig(
            shift_then_scale=True, share_a=shared_a, share_b=shared_b, device=device
        )
        shift_calibrator = TorchCalibrator(shift_config, n_topics, 4)
        shift_calibrator.fit(train_logits, train_labels, train_topics)
        _, _, shift_test_nce = evaluate_performance(test_logits, test_labels, test_topics_encoded, shift_calibrator)
        
        test_results.append({
            'Configuration': f"{config_name}, {b_name}",
            'Uncalibrated': test_uncal_nce,
            'a*logit+b': linear_test_nce,
            'a*(logit+b)': shift_test_nce
        })
    
    # Print all tables
    print("\n" + "="*80)
    print("FULL TRAINING SET RESULTS (Normalized Cross Entropy)")
    print("="*80)
    
    df_train = pd.DataFrame(full_train_results)
    print(df_train.to_string(index=False, float_format='%.4f'))
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS (Normalized Cross Entropy)")
    print("="*80)
    
    # Format CV results
    cv_formatted = []
    for i, result in enumerate(cv_results):
        cv_formatted.append({
            'Configuration': result['Configuration'],
            'Uncalibrated': f"{train_uncal_nce:.4f}",
            'a*logit+b': f"{result['Linear_Mean']:.4f} ± {result['Linear_SE']:.4f}",
            'a*(logit+b)': f"{result['Shift_Mean']:.4f} ± {result['Shift_SE']:.4f}"
        })
    
    df_cv = pd.DataFrame(cv_formatted)
    print(df_cv.to_string(index=False))
    
    print("\n" + "="*80)
    print("TEST SET RESULTS (Normalized Cross Entropy)")
    print("="*80)
    
    df_test = pd.DataFrame(test_results)
    print(df_test.to_string(index=False, float_format='%.4f'))
    
    return df_train, df_cv, df_test




###### PLOTS

def nce_from_logits_np(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-12, 1.0)
    ce = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    return ce / math.log(logits.shape[1])

def train_calibrator(train_logits, train_labels, train_topics, n_topics, device,
                     share_a, share_b, shift_then_scale):
    cfg = CalibrationConfig(shift_then_scale=shift_then_scale,
                            share_a=share_a, share_b=share_b,
                            device=device, dtype=torch.float64,
                            lr=1.0, max_iter=50, tol=1e-7)
    calib = TorchCalibrator(cfg, n_topics, 4)
    calib.fit(train_logits, train_labels, train_topics, verbose=False)
    return calib

def per_topic_breakdown(test_df, label_encoder, uncal_logits, uncal_labels,
                        trained_calibs, formula_name):
    subjects = list(label_encoder.classes_)
    rows = []
    for subj in subjects:
        mask = (test_df['subject'].values == subj)
        if not mask.any(): 
            continue
        logits_s = uncal_logits[mask]
        labels_s = uncal_labels[mask]
        topic_id = label_encoder.transform([subj])[0]
        topics_s = np.full(len(labels_s), topic_id, dtype=np.int64)

        scores = {'Uncalibrated': nce_from_logits_np(logits_s, labels_s)}
        for name, calib in trained_calibs.items():
            cal_logits = calib.predict_logits(logits_s, topics_s)
            scores[name] = nce_from_logits_np(cal_logits, labels_s)

        rows.append({'subject': subj, **scores})
    df = pd.DataFrame(rows).sort_values('subject').reset_index(drop=True)
    df.attrs['formula'] = formula_name
    return df

def plot_topic_bars(df, title, outfile):
    groups = ['Uncalibrated', 'Global', 'GlobalB_TopicA', 'FullTopic']
    x = np.arange(len(df)); width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(df)*0.4), 6))
    for i, g in enumerate(groups):
        ax.bar(x + (i-1.5)*width, df[g].values, width, label=g)
    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(df['subject'].values, rotation=90)
    ax.set_ylabel('NCE (lower is better)')
    ax.legend(); fig.tight_layout()
    plt.savefig(outfile, dpi=180); plt.close(fig)

def run_topic_breakdown(model_name="meta_llama_Llama_3.2_3B_Instruct"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_df, validation_df = load_mmlu_data(model_name)
    if test_df is None or validation_df is None:
        return None, None

    # train on MMLU "test", evaluate on "validation"
    train_logits, train_labels, train_topics, n_topics, label_encoder = prepare_data(test_df)
    test_logits, test_labels, _, _, _ = prepare_data(validation_df)

    # (1) a*score + b
    calibs_linear = {
        'Global':           train_calibrator(train_logits, train_labels, train_topics, n_topics, device, True,  True,  False),
        'GlobalB_TopicA':   train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, True,  False),
        'FullTopic':        train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, False, False),
    }
    df_linear = per_topic_breakdown(validation_df, label_encoder, test_logits, test_labels,
                                    calibs_linear, 'a*score+b')
    plot_topic_bars(df_linear, 'Per-topic NCE: a*score+b', 'nce_by_topic_linear.png')

    # (2) a*(score + b)
    calibs_shift = {
        'Global':           train_calibrator(train_logits, train_labels, train_topics, n_topics, device, True,  True,  True),
        'GlobalB_TopicA':   train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, True,  True),
        'FullTopic':        train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, False, True),
    }
    df_shift = per_topic_breakdown(validation_df, label_encoder, test_logits, test_labels,
                                   calibs_shift, 'a*(score+b)')
    plot_topic_bars(df_shift, 'Per-topic NCE: a*(score+b)', 'nce_by_topic_shift.png')

    # Print a quick summary focused on your hypothesis
    focus = df_shift[['subject', 'Uncalibrated', 'Global', 'GlobalB_TopicA', 'FullTopic']]
    print("\na*(score+b): per-topic NCE (lower is better)")
    print(focus.to_string(index=False, float_format='%.4f'))

    return df_linear, df_shift


if __name__ == "__main__":
    train_results, cv_results, test_results = run_mmlu_experiment()
    # Per-topic figures + table:
    _df_lin, _df_shift = run_topic_breakdown()





    train_results, cv_results, test_results = run_mmlu_experiment()
