import pandas as pd
from pathlib import Path
import json
from datasets import load_dataset

def download_and_process_mmlu(cache_dir='mmlu_cache'):
    """Download MMLU dataset from Hugging Face focusing on test, validation, and dev splits"""
    
    print("Downloading MMLU from Hugging Face...")
    
    try:
        # Load the full dataset
        dataset = load_dataset('cais/mmlu', 'all', cache_dir=cache_dir)
        print(f"Available splits: {list(dataset.keys())}")
        
        # Keep only the splits we want (exclude auxiliary_train)
        target_splits = {'test', 'validation', 'dev'}
        available_splits = set(dataset.keys())
        
        # Use intersection to get only splits that exist and we want
        splits_to_process = target_splits.intersection(available_splits)
        print(f"Processing splits: {list(splits_to_process)}")
        
        if not splits_to_process:
            print("Warning: None of the target splits (test, validation, dev) found!")
            print(f"Available splits are: {list(available_splits)}")
            # Fallback: use all available splits except auxiliary_train
            splits_to_process = available_splits - {'auxiliary_train'}
        
        all_data = {}
        
        for split_name in splits_to_process:
            print(f"\nProcessing {split_name} split...")
            split_data = dataset[split_name]
            print(f"  Split size: {len(split_data)}")
            
            # Group by subject
            subjects_data = {}
            
            for item in split_data:
                subject = item['subject']
                
                if subject not in subjects_data:
                    subjects_data[subject] = []
                
                subjects_data[subject].append({
                    'question': item['question'],
                    'options': item['choices'],
                    'correct_answer': item['choices'][item['answer']],
                    'correct_idx': item['answer']
                })
            
            all_data[split_name] = subjects_data
            
            # Print statistics
            total_questions = sum(len(questions) for questions in subjects_data.values())
            unique_subjects = len(subjects_data)
            print(f"  Total questions: {total_questions}")
            print(f"  Unique subjects: {unique_subjects}")
        
        return all_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        return None

def save_mmlu_processed(data, output_file='mmlu_processed.pkl'):
    """Save processed MMLU data to pickle file"""
    rows = []
    
    for split, split_data in data.items():
        for subject, questions in split_data.items():
            for q in questions:
                rows.append({
                    'split': split,
                    'category': subject,
                    'question': q['question'],
                    'options': q['options'],
                    'correct_answer': q['correct_answer'],
                    'correct_idx': q['correct_idx']
                })
    
    df = pd.DataFrame(rows)
    df.to_pickle(output_file)
    print(f"\nSaved {len(df)} questions to {output_file}")
    
    # Print summary statistics
    print("\nDataset summary:")
    print(f"Total questions: {len(df)}")
    print(f"Splits: {list(df['split'].unique())}")
    print(f"Number of subjects: {df['category'].nunique()}")
    
    print("\nQuestions per split:")
    for split in df['split'].value_counts().index:
        count = df['split'].value_counts()[split]
        print(f"  {split}: {count}")
    
    print("\nTop 10 subjects by question count:")
    print(df['category'].value_counts().head(10).to_string())
    
    return df

def save_sample_json(data, output_file='mmlu_sample.json', max_examples=100):
    """Save a small sample as JSON for inspection"""
    rows = []
    
    for split, split_data in data.items():
        for subject, questions in split_data.items():
            for q in questions[:2]:  # Max 2 per subject per split
                rows.append({
                    'split': split,
                    'category': subject,
                    'question': q['question'],
                    'options': q['options'],
                    'correct_answer': q['correct_answer'],
                    'correct_idx': q['correct_idx']
                })
                
                if len(rows) >= max_examples:
                    break
            if len(rows) >= max_examples:
                break
        if len(rows) >= max_examples:
            break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(rows)} sample questions to {output_file}")

def verify_data(df):
    """Verify the dataset structure"""
    print("\n" + "="*50)
    print("DATASET VERIFICATION")
    print("="*50)
    
    # Check splits
    splits = set(df['split'].unique())
    expected_splits = {'test', 'validation', 'dev'}
    
    print(f"Found splits: {splits}")
    print(f"Expected splits: {expected_splits}")
    
    if expected_splits.issubset(splits):
        print("✓ All expected splits present")
    else:
        missing = expected_splits - splits
        extra = splits - expected_splits
        if missing:
            print(f"⚠ Missing splits: {missing}")
        if extra:
            print(f"ℹ Extra splits found: {extra}")
    
    # Check subjects
    n_subjects = df['category'].nunique()
    print(f"\nSubjects found: {n_subjects}")
    
    # Show split distribution
    print(f"\nSplit distribution:")
    for split in sorted(df['split'].unique()):
        count = len(df[df['split'] == split])
        subjects_in_split = df[df['split'] == split]['category'].nunique()
        print(f"  {split}: {count} questions across {subjects_in_split} subjects")
    
    print("="*50)

def show_examples(df, n_examples=1):
    """Show example questions from each split"""
    print(f"\nExample questions:")
    
    for split in sorted(df['split'].unique()):
        split_data = df[df['split'] == split]
        print(f"\n--- {split.upper()} SPLIT ---")
        
        for i in range(min(n_examples, len(split_data))):
            example = split_data.iloc[i]
            print(f"Subject: {example['category']}")
            print(f"Q: {example['question']}")
            print("Options:")
            for j, opt in enumerate(example['options']):
                marker = "✓" if j == example['correct_idx'] else " "
                print(f"  {marker} {chr(65+j)}) {opt}")
            print()

if __name__ == "__main__":
    print("Starting simplified MMLU download...")
    print("Note: This script excludes 'auxiliary_train' and focuses on test, validation, and dev splits")
    
    # Download and process
    data = download_and_process_mmlu()
    
    if data is None:
        print("Failed to download MMLU dataset.")
        exit(1)
    
    # Save processed data
    df = save_mmlu_processed(data)
    save_sample_json(data)
    
    # Verify and show examples
    verify_data(df)
    show_examples(df)
    
    print(f"\n✓ Successfully processed MMLU dataset!")
    print(f"  - Saved full dataset to: mmlu_processed.pkl")
    print(f"  - Saved sample to: mmlu_sample.json")
    print(f"  - Total questions: {len(df)}")
    print(f"  - Subjects: {df['category'].nunique()}")
    print(f"  - Splits: {', '.join(sorted(df['split'].unique()))}")

