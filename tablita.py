import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
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
        self.line_search_fn = None
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
        device = torch.device('cpu')
        logits = torch.from_numpy(logits_np).to(device=device, dtype=torch.float64)
        targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
        topics = torch.from_numpy(topics_np).to(device=device, dtype=torch.long)

        self.to(device=device, dtype=torch.float64)

        with torch.no_grad():
            initial_loss = F.cross_entropy(self.forward(logits, topics), targets).item()

        with torch.no_grad():
            p0 = torch.cat([p.detach().flatten().clone() for p in self.parameters()])

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=self.config.lr,
            max_iter=self.config.max_iter,
            history_size=self.config.history_size,
            tolerance_grad=self.config.tol,
            tolerance_change=self.config.tol,
        )

        it = {'k': 0}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            out = self.forward(logits, topics)
            loss = F.cross_entropy(out, targets)
            loss.backward()
            if verbose and it['k'] in (0, 1):
                gnorm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in self.parameters() if p.grad is not None)).item()
                print(f"    closure {it['k']}: loss={loss.item():.6f}, ||grad||={gnorm:.3e}")
            it['k'] += 1
            return loss

        final_loss_returned = optimizer.step(closure)

        optimizer.zero_grad(set_to_none=True)
        out_final = self.forward(logits, topics)
        final_loss_tensor = F.cross_entropy(out_final, targets)
        final_loss_eval = final_loss_tensor.item()
        final_loss_tensor.backward()
        grad_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in self.parameters() if p.grad is not None)).item()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            p1 = torch.cat([p.detach().flatten() for p in self.parameters()])
            delta = p1 - p0
            dnorm = delta.norm().item()

        if verbose:
            returned = final_loss_returned.item() if isinstance(final_loss_returned, torch.Tensor) else float(final_loss_returned)
            improvement = initial_loss - final_loss_eval
            grad_tol = max(self.config.tol, 1e-9)
            param_tol = max(self.config.tol, 1e-9)
            loss_tol = max(self.config.tol, 1e-9)
            converged = (grad_norm <= grad_tol) or (dnorm <= param_tol and abs(improvement) <= loss_tol)
            print(
                f"  LBFGS finished, loss: {final_loss_eval:.6f}, returned: {returned:.6f}, ||Δparam||={dnorm:.3e}, "
                f"Δloss={improvement:.3e}, final||grad||={grad_norm:.3e}, closures={it['k']}, converged={converged}"
            )

        self.device = device
        self.dtype = torch.float64

    def predict_logits(self, logits_np, topics_np):
        device = getattr(self, 'device', torch.device(self.config.device))
        dtype = getattr(self, 'dtype', self.config.dtype)
        logits = torch.from_numpy(logits_np).to(device=device, dtype=dtype)
        topics = torch.from_numpy(topics_np).to(device=device, dtype=torch.long)
        with torch.no_grad():
            out = self.forward(logits, topics)
        return out.cpu().numpy()

    def predict_zero_scale_logits(self, logits_np, topics_np):
        device = getattr(self, 'device', torch.device(self.config.device))
        dtype = getattr(self, 'dtype', self.config.dtype)
        logits = torch.from_numpy(logits_np).to(device=device, dtype=dtype)
        topics = torch.from_numpy(topics_np).to(device=device, dtype=torch.long)

        with torch.no_grad():
            if self.config.share_b:
                bias = self.b.unsqueeze(0).expand(logits.shape[0], -1)
            else:
                bias = self.b[topics]

            if self.config.shift_then_scale:
                zeros = torch.zeros_like(logits)
                return zeros.cpu().numpy()
            else:
                return bias.cpu().numpy()

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

def prepare_data(df, label_encoder=None, fit_encoder=True):
    """Prepare data for calibration"""
    logit_cols = ['logit_A', 'logit_B', 'logit_C', 'logit_D']
    logits = df[logit_cols].values
    labels = df['correct_answer_position'].values
    question_ids = df['question_id'].values if 'question_id' in df.columns else None
    subjects = df['subject'].values

    if fit_encoder:
        label_encoder = LabelEncoder()
        topics = label_encoder.fit_transform(subjects)
    else:
        if label_encoder is None:
            raise ValueError("label_encoder must be provided when fit_encoder is False")
        try:
            topics = label_encoder.transform(subjects)
        except ValueError:
            mapping = {s: i for i, s in enumerate(label_encoder.classes_)}
            fallback_id = 0
            unseen = sorted(set(subjects) - set(label_encoder.classes_))
            if unseen:
                print(f"[warn] {len(unseen)} unseen subject(s) in this split; falling back to topic {fallback_id}")
            topics = np.array([mapping.get(s, fallback_id) for s in subjects], dtype=np.int64)

    unique_subjects = len(np.unique(subjects))

    print("Data preparation:")
    print(f"  Samples: {len(logits)}")
    print(f"  Unique subjects: {unique_subjects}")
    print(f"  Logits shape: {logits.shape}")

    return logits, labels, topics, unique_subjects, label_encoder, question_ids

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

def run_calibration_cv(logits, labels, topics, n_topics, device='cpu', n_folds=5, question_ids=None):
    """Run cross-validation calibration experiment grouped by question id"""

    configs = [
        ('a_shared', 'b_shared', True, True),
        ('a_topic', 'b_shared', False, True),
        ('a_topic', 'b_topic', False, False),
    ]

    if question_ids is None:
        if len(logits) % 24 != 0:
            raise ValueError("Question ids required when rows per question are not 24")
        question_ids = np.repeat(np.arange(len(logits) // 24), 24)

    question_ids = np.asarray(question_ids)

    splitter = GroupKFold(n_splits=n_folds)
    cv_results = []

    def mean_and_se(values):
        arr = np.asarray(values, dtype=np.float64)
        mean = arr.mean()
        if arr.size <= 1:
            return mean, 0.0
        return mean, arr.std(ddof=1) / np.sqrt(arr.size)

    for config_name, b_name, shared_a, shared_b in configs:
        print(f"Cross-validation for {config_name}, {b_name}...")

        uncal_nces = []
        linear_nces = []
        shift_nces = []

        for fold, (train_idx, val_idx) in enumerate(splitter.split(logits, labels, groups=question_ids)):
            train_logits = logits[train_idx]
            train_labels = labels[train_idx]
            train_topics = topics[train_idx]

            val_logits = logits[val_idx]
            val_labels = labels[val_idx]
            val_topics = topics[val_idx]

            uncal_nces.append(nce_from_logits_np(val_logits, val_labels))

            linear_config = CalibrationConfig(
                shift_then_scale=False, share_a=shared_a, share_b=shared_b, device=device
            )
            linear_calibrator = TorchCalibrator(linear_config, n_topics, 4)
            linear_calibrator.fit(train_logits, train_labels, train_topics)
            _, _, linear_nce = evaluate_performance(val_logits, val_labels, val_topics, linear_calibrator)
            linear_nces.append(linear_nce)

            shift_config = CalibrationConfig(
                shift_then_scale=True, share_a=shared_a, share_b=shared_b, device=device
            )
            shift_calibrator = TorchCalibrator(shift_config, n_topics, 4)
            shift_calibrator.fit(train_logits, train_labels, train_topics)
            _, _, shift_nce = evaluate_performance(val_logits, val_labels, val_topics, shift_calibrator)
            shift_nces.append(shift_nce)

        uncal_mean, uncal_se = mean_and_se(uncal_nces)
        linear_mean, linear_se = mean_and_se(linear_nces)
        shift_mean, shift_se = mean_and_se(shift_nces)

        cv_results.append({
            'Configuration': f"{config_name}, {b_name}",
            'Uncal_Mean': uncal_mean,
            'Uncal_SE': uncal_se,
            'Linear_Mean': linear_mean,
            'Linear_SE': linear_se,
            'Shift_Mean': shift_mean,
            'Shift_SE': shift_se
        })

    return cv_results

def run_mmlu_experiment():
    """Run the complete MMLU calibration experiment"""
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
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
    train_logits, train_labels, train_topics, n_topics, label_encoder, train_question_ids = prepare_data(test_df)
    # Use "validation" set for final testing
    test_logits, test_labels, test_topics, _, _, _ = prepare_data(validation_df, label_encoder=label_encoder, fit_encoder=False)
    
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
        lin_logits = linear_calibrator.predict_logits(train_logits, train_topics)
        lin_train_ce = cross_entropy_from_logits_np(lin_logits, train_labels)
        print(f"    train CE (a*logit+b): {train_uncal_ce:.6f} → {lin_train_ce:.6f}")
        zero_scale_ce_lin, zero_warnings_lin = zero_scale_analysis(
            linear_calibrator,
            train_logits,
            train_topics,
            train_labels,
            label_encoder,
            calibrated_logits=lin_logits,
        )
        msg = "    zero-scale CE (a=0): " + f"{zero_scale_ce_lin:.6f}"
        if zero_warnings_lin:
            msg += "  [warn topic-level improvements detected]"
        print(msg)
        for topic_id, subject, ce_cal, ce_zero in zero_warnings_lin:
            label_str = f" ({subject})" if subject is not None else ""
            print(
                f"      topic {int(topic_id)}{label_str}: zero-scale {ce_zero:.6f} < calibrated {ce_cal:.6f}"
            )
        _, _, linear_nce = evaluate_performance(train_logits, train_labels, train_topics, linear_calibrator)
        
        # Shift-scale calibration
        shift_config = CalibrationConfig(
            shift_then_scale=True, share_a=shared_a, share_b=shared_b, device=device
        )
        shift_calibrator = TorchCalibrator(shift_config, n_topics, 4)
        shift_calibrator.fit(train_logits, train_labels, train_topics, verbose=True)
        shift_logits = shift_calibrator.predict_logits(train_logits, train_topics)
        shift_train_ce = cross_entropy_from_logits_np(shift_logits, train_labels)
        print(f"    train CE (a*(logit+b)): {train_uncal_ce:.6f} → {shift_train_ce:.6f}")
        zero_scale_ce_shift, zero_warnings_shift = zero_scale_analysis(
            shift_calibrator,
            train_logits,
            train_topics,
            train_labels,
            label_encoder,
            calibrated_logits=shift_logits,
        )
        msg_shift = "    zero-scale CE (a=0): " + f"{zero_scale_ce_shift:.6f}"
        if zero_warnings_shift:
            msg_shift += "  [warn topic-level improvements detected]"
        print(msg_shift)
        for topic_id, subject, ce_cal, ce_zero in zero_warnings_shift:
            label_str = f" ({subject})" if subject is not None else ""
            print(
                f"      topic {int(topic_id)}{label_str}: zero-scale {ce_zero:.6f} < calibrated {ce_cal:.6f}"
            )
        _, _, shift_nce = evaluate_performance(train_logits, train_labels, train_topics, shift_calibrator)
        
        full_train_results.append({
            'Configuration': f"{config_name}, {b_name}",
            'Uncalibrated': train_uncal_nce,
            'a*logit+b': linear_nce,
            'a*(logit+b)': shift_nce
        })
    
    # 2. Cross-validation results
    print("\nRunning cross-validation...")
    cv_results = run_calibration_cv(train_logits, train_labels, train_topics, n_topics, device, question_ids=train_question_ids)
    
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
        _, _, linear_test_nce = evaluate_performance(test_logits, test_labels, test_topics, linear_calibrator)
        
        shift_config = CalibrationConfig(
            shift_then_scale=True, share_a=shared_a, share_b=shared_b, device=device
        )
        shift_calibrator = TorchCalibrator(shift_config, n_topics, 4)
        shift_calibrator.fit(train_logits, train_labels, train_topics)
        _, _, shift_test_nce = evaluate_performance(test_logits, test_labels, test_topics, shift_calibrator)
        
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
    for result in cv_results:
        cv_formatted.append({
            'Configuration': result['Configuration'],
            'Uncalibrated': f"{result['Uncal_Mean']:.4f} ± {result['Uncal_SE']:.4f}",
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

def cross_entropy_from_logits_np(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels]))


def nce_from_logits_np(logits, labels):
    ce = cross_entropy_from_logits_np(logits, labels)
    return ce / math.log(logits.shape[1])


def zero_scale_analysis(calibrator, logits, topics, labels, label_encoder, calibrated_logits=None, eps=1e-9):
    if calibrated_logits is None:
        calibrated_logits = calibrator.predict_logits(logits, topics)

    zero_logits = calibrator.predict_zero_scale_logits(logits, topics)
    zero_ce = cross_entropy_from_logits_np(zero_logits, labels)

    warnings = []
    unique_topics = np.unique(topics)
    for topic_id in unique_topics:
        mask = (topics == topic_id)
        if mask.sum() == 0:
            continue
        ce_calibrated = cross_entropy_from_logits_np(calibrated_logits[mask], labels[mask])
        ce_zero = cross_entropy_from_logits_np(zero_logits[mask], labels[mask])
        if ce_zero + eps < ce_calibrated:
            subject = None
            if label_encoder is not None and 0 <= topic_id < len(label_encoder.classes_):
                try:
                    subject = label_encoder.inverse_transform([topic_id])[0]
                except Exception:
                    subject = None
            warnings.append((topic_id, subject, ce_calibrated, ce_zero))

    return zero_ce, warnings

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
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    test_df, validation_df = load_mmlu_data(model_name)
    if test_df is None or validation_df is None:
        return None, None

    # train on MMLU "test", evaluate on "validation"
    train_logits, train_labels, train_topics, n_topics, label_encoder, _ = prepare_data(test_df)
    test_logits = validation_df[['logit_A', 'logit_B', 'logit_C', 'logit_D']].values
    test_labels = validation_df['correct_answer_position'].values

    # (1) a*score + b
    calibs_linear = {
        'Global':           train_calibrator(train_logits, train_labels, train_topics, n_topics, device, True,  True,  False),
        'GlobalB_TopicA':   train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, True,  False),
        'FullTopic':        train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, False, False),
    }
    df_linear = per_topic_breakdown(validation_df, label_encoder, test_logits, test_labels,
                                    calibs_linear, 'a*score+b')
    plot_topic_bars(df_linear, 'Per-topic NCE: a*score+b', 'nce_by_topic_linear.png')
    df_linear.to_csv('nce_by_topic_linear.csv', index=False)

    # (2) a*(score + b)
    calibs_shift = {
        'Global':           train_calibrator(train_logits, train_labels, train_topics, n_topics, device, True,  True,  True),
        'GlobalB_TopicA':   train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, True,  True),
        'FullTopic':        train_calibrator(train_logits, train_labels, train_topics, n_topics, device, False, False, True),
    }
    df_shift = per_topic_breakdown(validation_df, label_encoder, test_logits, test_labels,
                                   calibs_shift, 'a*(score+b)')
    plot_topic_bars(df_shift, 'Per-topic NCE: a*(score+b)', 'nce_by_topic_shift.png')
    df_shift.to_csv('nce_by_topic_shift.csv', index=False)

    # Print a quick summary focused on your hypothesis
    focus = df_shift[['subject', 'Uncalibrated', 'Global', 'GlobalB_TopicA', 'FullTopic']]
    print("\na*(score+b): per-topic NCE (lower is better)")
    print(focus.to_string(index=False, float_format='%.4f'))

    return df_linear, df_shift


if __name__ == "__main__":
    train_results, cv_results, test_results = run_mmlu_experiment()
    # Per-topic figures + table:
    _df_lin, _df_shift = run_topic_breakdown()
