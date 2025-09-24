import os
from typing import Dict, List, Optional

import torch
import numpy as np
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import pickle
import hashlib
import pandas as pd
import itertools
import gc

# Load environment variables
load_dotenv()
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)

# Model selection - uncomment the one you want to use
# model_id = "meta-llama/Llama-3.2-1B-Instruct"  # 1B parameters (default)
model_id = "meta-llama/Llama-3.2-3B-Instruct"  # 3B parameters
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # 7B parameters
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # 8B parameters

print(f"Loading model: {model_id}")
print("This may take a while for larger models...")

# Load tokenizer first
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model with memory-efficient settings
print("Loading model with automatic device mapping...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
    device_map="auto",  # Automatically distribute across available GPUs/CPU
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)

print(f"Model loaded! Device map: {model.hf_device_map}")


def _resolve_module_from_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """Resolve a dotted path (allowing integer indices) into a module reference."""
    module = root
    if not path:
        raise ValueError("Empty module path provided for probe registration")

    for segment in path.split('.'):
        if segment == "":
            raise ValueError(f"Invalid segment in module path '{path}'")

        if segment.isdigit():
            try:
                module = module[int(segment)]
            except (TypeError, IndexError) as exc:
                raise AttributeError(
                    f"Module '{module.__class__.__name__}' does not support integer access at segment '{segment}' (path '{path}')"
                ) from exc
        else:
            if not hasattr(module, segment):
                raise AttributeError(f"Module '{module.__class__.__name__}' has no attribute '{segment}' (while resolving '{path}')")
            module = getattr(module, segment)

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Resolved object for path '{path}' is not a torch.nn.Module")

    return module


def _to_storage_dtype(tensor: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """Convert tensor to the requested storage dtype on CPU."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    target_dtype = dtype_map.get(dtype_str.lower())
    if target_dtype is None:
        raise ValueError(f"Unsupported probe storage dtype '{dtype_str}'. Supported values: {list(dtype_map)}")

    return tensor.to(device="cpu", dtype=target_dtype)


class ActivationProbeManager:
    """Register forward hooks to capture intermediate activations for analysis."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_paths: Optional[List[str]] = None,
        storage_dtype: str = "float32",
        capture_all_forwards: bool = False,
    ) -> None:
        self.model = model
        self.layer_paths = layer_paths or []
        self.storage_dtype = storage_dtype
        self.capture_all_forwards = capture_all_forwards

        self.enabled = bool(self.layer_paths)
        self._recording = False
        self._current_key: Optional[str] = None
        self._handles: List[RemovableHandle] = []
        self._records: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self._memory_bytes: Dict[str, int] = {}

        if self.enabled:
            self._register_hooks()

    def _register_hooks(self) -> None:
        for path in self.layer_paths:
            module = _resolve_module_from_path(self.model, path)
            handle = module.register_forward_hook(self._make_hook(path))
            self._handles.append(handle)
            print(f"[probe] Registered hook on module: {path}")

    def _make_hook(self, layer_path: str):
        def hook(_: torch.nn.Module, __, output):
            if not self._recording or self._current_key is None:
                return

            tensor = self._extract_tensor(output)
            if tensor is None:
                return

            stored_tensor = _to_storage_dtype(tensor, self.storage_dtype)

            layer_records = self._records.setdefault(self._current_key, {}).setdefault(layer_path, [])
            layer_records.append(stored_tensor)

            bytes_used = stored_tensor.element_size() * stored_tensor.nelement()
            self._memory_bytes[self._current_key] = self._memory_bytes.get(self._current_key, 0) + bytes_used

        return hook

    def _extract_tensor(self, output) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output.detach()
        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first.detach()
        return None

    def start_recording(self, key: str) -> None:
        if not self.enabled:
            return
        self._current_key = key
        self._recording = True
        # Reset records for this key to avoid mixing runs
        self._records[key] = {}
        self._memory_bytes[key] = 0

    def stop_recording(self) -> None:
        self._recording = False
        self._current_key = None

    def should_record(self, explicit_request: bool) -> bool:
        return self.enabled and (self.capture_all_forwards or explicit_request)

    def has_recordings(self, key: str) -> bool:
        return key in self._records and any(self._records[key].values())

    def get_recordings(self, key: str) -> Optional[Dict[str, List[torch.Tensor]]]:
        if not self.enabled:
            return None
        return self._records.get(key)

    def get_memory_usage(self, key: Optional[str] = None) -> Dict[str, int]:
        if key is not None:
            return {key: self._memory_bytes.get(key, 0)}
        return dict(self._memory_bytes)

    def clear_recordings(self) -> None:
        self._records.clear()
        self._memory_bytes.clear()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


probe_layer_spec = os.getenv("PROBE_LAYER_PATHS", "")
probe_layers = [segment.strip() for segment in probe_layer_spec.split(',') if segment.strip()]
probe_storage_dtype = os.getenv("PROBE_STORAGE_DTYPE", "float32")
probe_capture_all = os.getenv("PROBE_CAPTURE_ALL", "0") == "1"

probe_manager = ActivationProbeManager(
    model=model,
    layer_paths=probe_layers,
    storage_dtype=probe_storage_dtype,
    capture_all_forwards=probe_capture_all,
)

# Get token IDs for answer options
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
letter_ids = [tokenizer.convert_tokens_to_ids(letter) for letter in letters]

# Model-specific cache file
model_name_clean = model_id.replace("/", "_").replace("-", "_")
cache_filename = f'forward_cache_{model_name_clean}.pkl'

def save_cache_safely(cache, cache_filename):
    """Save cache with backup to prevent corruption"""
    temp_filename = cache_filename + '.tmp'
    
    # Save to temporary file first
    with open(temp_filename, 'wb') as f:
        pickle.dump(cache, f)
    
    # Only replace the original if temp save succeeded
    if os.path.exists(cache_filename):
        os.rename(cache_filename, cache_filename + '.bak')
    os.rename(temp_filename, cache_filename)
    
    # Clean up backup after successful save
    backup_file = cache_filename + '.bak'
    if os.path.exists(backup_file):
        os.remove(backup_file)

def load_cache_safely(cache_filename):
    """Load cache with fallback to backup if main file is corrupted"""
    cache = {}
    
    # Try loading main cache file
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded cache with {len(cache)} entries from {cache_filename}")
            return cache
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache file corrupted: {e}")
            
            # Try backup file
            backup_file = cache_filename + '.bak'
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'rb') as f:
                        cache = pickle.load(f)
                    print(f"Loaded cache from backup with {len(cache)} entries")
                    return cache
                except (EOFError, pickle.UnpicklingError):
                    print("Backup cache also corrupted, starting fresh")
            else:
                print("No backup available, starting fresh")
    
    print(f"Starting with empty cache (will save to {cache_filename})")
    return cache

# Load cache safely
cache = load_cache_safely(cache_filename)

def hash_string(s: str) -> str:
    """Hash a string for cache keys"""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def forward(prompt, nopts=4, *, return_activations: bool = False, probe_id: Optional[str] = None):
    """
    Compute raw logits for the answer tokens without any masking.

    When ``return_activations`` is True and probe layers are configured, this function
    also returns the captured activations for the requested layers.
    """
    key = hash_string(prompt)
    sample_key = probe_id or key
    cache_hit = key in cache
    need_fresh_activations = (
        return_activations
        and probe_manager.enabled
        and not probe_manager.has_recordings(sample_key)
    )

    if cache_hit and not need_fresh_activations:
        if return_activations:
            activations = probe_manager.get_recordings(sample_key) if probe_manager.enabled else None
            return cache[key], activations
        return cache[key]

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    record_activations = probe_manager.should_record(return_activations)
    if record_activations:
        probe_manager.start_recording(sample_key)

    # Forward pass to get logits
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits for the last token position
            last_token_logits = outputs.logits[0, -1, :]

            # Extract logits for the letter tokens
            scores = last_token_logits[letter_ids[:nopts]]
    finally:
        if record_activations:
            probe_manager.stop_recording()

    # Cache the result
    cache[key] = scores

    # Save cache periodically (every 20 new entries)
    if len(cache) % 20 == 0:
        save_cache_safely(cache, cache_filename)
        # print(f"Cache saved ({len(cache)} entries)")

    if return_activations:
        activations = probe_manager.get_recordings(sample_key) if probe_manager.enabled else None
        return scores, activations

    return scores

def write_prompt(question, options, sep=')'):
    """Format a multiple choice question as a prompt"""
    # Adjust prompt style based on model
    if "mistral" in model_id.lower():
        # Mistral format
        system_prompt = "[INST] Answer the following multiple choice question by responding with just the letter of the correct answer."
        prompt = f"{system_prompt}\n\n{question}\n"
        for i, val in enumerate(options):
            prompt += f"{letters[i]}{sep} {val}\n"
        prompt += "\nAnswer: [/INST]"
    else:
        # Llama format
        system_prompt = "Answer the following multiple choice question by responding with just the letter of the correct answer."
        prompt = f"{system_prompt}\n\n{question}\n"
        for i, val in enumerate(options):
            prompt += f"{letters[i]}{sep} {val}\n"
        prompt += "Answer: "
    
    return prompt

def generate_question_id(row_idx, question, subject):
    """Generate a unique question ID"""
    # Create a short hash of the question text for uniqueness
    question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()[:8]
    return f"{subject}_{row_idx:04d}_{question_hash}"

def process_mmlu_split(df, split_name, num_questions=None, random_sample=False):
    """Process MMLU questions for a specific split and generate permutation dataset"""
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    
    # Filter to the specific split
    split_df = df[df['split'] == split_name].copy()
    
    if len(split_df) == 0:
        print(f"No questions found for {split_name} split!")
        return None
    
    print(f"Total questions in {split_name}: {len(split_df)}")
    
    # Sample questions if specified
    if num_questions is not None:
        if random_sample:
            split_df = split_df.sample(n=min(num_questions, len(split_df)), random_state=42)
            print(f"Randomly sampled {len(split_df)} questions")
        else:
            split_df = split_df.head(num_questions)
            print(f"Taking first {len(split_df)} questions")
    
    results = []
    total = len(split_df)
    
    for idx, (_, row) in enumerate(split_df.iterrows()):
        if idx % 5 == 0:
            print(f"Processing question {idx+1}/{total}... (Cache size: {len(cache)})")
            # Clear CUDA cache periodically for larger models
            if torch.cuda.is_available() and idx % 10 == 0:
                torch.cuda.empty_cache()
        
        question = row['question']
        options = row['options']
        correct_idx = row['correct_idx']
        subject = row['category']
        
        # Handle options format - ensure it's a list
        if isinstance(options, str):
            # If options are stored as string representation, eval them
            try:
                options = eval(options)
            except:
                print(f"Warning: Could not parse options for question {idx}: {options}")
                continue
        
        # Generate unique question ID
        question_id = generate_question_id(idx, question, subject)
        
        # Generate all permutations (24 total for 4 options)
        for perm_idx, perm in enumerate(itertools.permutations(range(len(options)))):
            perm_options = [options[i] for i in perm]
            prompt = write_prompt(question, perm_options)
            scores = forward(prompt, len(options))
            
            # Find where the correct answer ended up after permutation
            new_correct_idx = perm.index(correct_idx)
            
            # Convert permutation to string representation (e.g., "ABCD", "ACBD")
            perm_str = ''.join([letters[i] for i in perm[:len(options)]])
            
            # Convert scores to float32 first to avoid BFloat16 issues
            scores_float = scores.float().cpu().numpy()
            predicted_idx = int(torch.argmax(scores).cpu().item())
            
            results.append({
                'question_id': question_id,
                'subject': subject,
                'split': split_name,
                'permutation_idx': perm_idx,
                'permutation': perm_str,
                'correct_answer_position': new_correct_idx,
                'logit_A': float(scores_float[0]) if len(scores_float) > 0 else None,
                'logit_B': float(scores_float[1]) if len(scores_float) > 1 else None,
                'logit_C': float(scores_float[2]) if len(scores_float) > 2 else None,
                'logit_D': float(scores_float[3]) if len(scores_float) > 3 else None,
                'predicted_answer': predicted_idx,
                'is_correct': predicted_idx == new_correct_idx,
                'original_question': question,
                'original_options': str(options),  # Store as string for CSV compatibility
                'permuted_options': str(perm_options)
            })
    
    # Save cache at the end
    save_cache_safely(cache, cache_filename)
    print(f"Cache saved after processing {split_name} ({len(cache)} entries)")
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Generate filename with model name and split
    filename = f"mmlu_logits_{model_name_clean}_{split_name}.csv"
    
    results_df.to_csv(filename, index=False)
    print(f"Saved {len(results_df)} rows to {filename}")
    print(f"Note: File will be overwritten if it exists")
    
    # Show summary statistics
    total_questions = len(results_df) // 24  # 24 permutations per question
    correct_predictions = results_df['is_correct'].sum()
    accuracy = correct_predictions / len(results_df) * 100
    
    print(f"\n{split_name.upper()} SPLIT SUMMARY:")
    print(f"Questions processed: {total_questions}")
    print(f"Total rows (24 permutations × questions): {len(results_df)}")
    print(f"Correct predictions: {correct_predictions}/{len(results_df)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Subjects covered: {results_df['subject'].nunique()}")
    
    return results_df

def display_sample_results(results_df, split_name, num_samples=3):
    """Display a sample of results for verification"""
    print(f"\n{'='*80}")
    print(f"SAMPLE RESULTS FROM {split_name.upper()} SPLIT")
    print(f"{'='*80}")
    
    # Get unique question IDs and sample a few
    unique_questions = results_df['question_id'].unique()
    sample_questions = unique_questions[:num_samples]
    
    for q_id in sample_questions:
        question_data = results_df[results_df['question_id'] == q_id]
        first_row = question_data.iloc[0]
        
        print(f"\nQuestion ID: {q_id}")
        print(f"Subject: {first_row['subject']}")
        print(f"Question: {first_row['original_question'][:100]}...")
        
        # Handle options display
        try:
            original_options = eval(first_row['original_options'])
            print(f"Original Options: {original_options}")
        except:
            print(f"Original Options: {first_row['original_options']}")
        
        print(f"Correct Answer: {letters[first_row['correct_answer_position']]} (position {first_row['correct_answer_position']})")
        
        print(f"\nPermutation samples (showing first 3 of 24):")
        print(f"{'Perm':<6} {'A':<8} {'B':<8} {'C':<8} {'D':<8} {'Pred':<6} {'Correct':<8}")
        print("-" * 50)
        
        for idx in range(min(3, len(question_data))):
            row = question_data.iloc[idx]
            pred_letter = letters[row['predicted_answer']]
            correct_mark = "✓" if row['is_correct'] else "✗"
            
            print(f"{row['permutation']:<6} {row['logit_A']:<8.2f} {row['logit_B']:<8.2f} {row['logit_C']:<8.2f} {row['logit_D']:<8.2f} {pred_letter:<6} {correct_mark:<8}")
        
        if len(question_data) > 3:
            print(f"... and {len(question_data) - 3} more permutations")
        print("-" * 80)

def main():
    """Main function to run the script"""
    print("MMLU Permutation Dataset Generator")
    print("="*50)
    
    # Check available memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {gpu_mem:.1f} GB")
    else:
        print("No GPU available, using CPU (this will be slow for large models!)")
    
    # Check if MMLU data exists
    if not os.path.exists("mmlu_processed.pkl"):
        print("ERROR: mmlu_processed.pkl not found!")
        print("Please run the simplified MMLU download script first.")
        return
    
    # Load MMLU data
    print("\nLoading MMLU data...")
    df = pd.read_pickle("mmlu_processed.pkl")
    print(f"Loaded {len(df)} questions")
    
    # Show split distribution
    print("\nSplit distribution:")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} questions")
    
    # Process only the splits we want (test, validation, dev)
    target_splits = ['test', 'validation', 'dev']
    available_splits = [split for split in target_splits if split in df['split'].unique()]
    
    if not available_splits:
        print("Warning: None of the target splits (test, validation, dev) found!")
        print("Available splits:", df['split'].unique())
        # Fall back to whatever splits are available
        available_splits = df['split'].unique()
    
    print(f"\nProcessing splits: {available_splits}")
    
    # Test mode: process 10 questions per split
    print(f"\n🧪 RUNNING TEST MODE: Processing 10 questions per split")
    print("Set num_questions=None for full processing")
    
    results = {}
    for split in available_splits:
        print(f"\n⏳ Starting {split} split...")
        split_results = process_mmlu_split(
            df, 
            split, 
            num_questions=None,  # Test with 10 questions
            random_sample=False  # Random sample
        )
        if split_results is not None:
            results[split] = split_results
            display_sample_results(split_results, split, num_samples=2)
    
    print(f"\n{'='*80}")
    print("🎉 PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Model: {model_id}")
    print(f"Files generated: {len(results)}")
    for split, df_result in results.items():
        total_questions = len(df_result) // 24
        accuracy = df_result['is_correct'].mean() * 100
        print(f"  {split}: {total_questions} questions, {len(df_result)} rows, {accuracy:.1f}% accuracy")
    
    print(f"\nCache file: {cache_filename}")
    print("To process full dataset, set num_questions=None in process_mmlu_split calls")

if __name__ == "__main__":
    main()
