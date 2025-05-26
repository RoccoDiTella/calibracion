import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def find_forward_files():
    """Find all MMLU forward pass CSV files"""
    pattern = "mmlu_logits_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No MMLU forward pass files found!")
        print("Expected files matching pattern: mmlu_logits_*.csv")
        return []
    
    print(f"✅ Found {len(files)} forward pass files:")
    for f in sorted(files):
        print(f"  - {f}")
    return sorted(files)

def verify_file_structure(df, filename):
    """Verify the basic structure of a forward pass file"""
    print(f"\n{'='*60}")
    print(f"VERIFYING: {filename}")
    print(f"{'='*60}")
    
    required_columns = [
        'question_id', 'subject', 'split', 'permutation_idx', 'permutation',
        'correct_answer_position', 'logit_A', 'logit_B', 'logit_C', 'logit_D',
        'predicted_answer', 'is_correct', 'original_question', 'original_options', 'permuted_options'
    ]
    
    # Check columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    else:
        print("✅ All required columns present")
    
    # Check basic stats
    print(f"📊 Dataset size: {len(df):,} rows")
    print(f"📝 Unique questions: {df['question_id'].nunique():,}")
    print(f"📚 Subjects: {df['subject'].nunique()}")
    print(f"🔄 Permutations per question: {len(df) // df['question_id'].nunique()}")
    
    # Check for missing values in critical columns
    critical_cols = ['logit_A', 'logit_B', 'logit_C', 'logit_D', 'predicted_answer', 'is_correct']
    missing_data = {}
    for col in critical_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_data[col] = missing_count
    
    if missing_data:
        print(f"⚠️  Missing values found:")
        for col, count in missing_data.items():
            print(f"   {col}: {count} missing values")
    else:
        print("✅ No missing values in critical columns")
    
    return True

def verify_permutations(df):
    """Verify that permutations are correct"""
    print(f"\n🔄 PERMUTATION VERIFICATION")
    print("-" * 40)
    
    # Check if we have exactly 24 permutations per question
    perm_counts = df.groupby('question_id')['permutation_idx'].count()
    expected_perms = 24  # 4! = 24 permutations
    
    incorrect_counts = perm_counts[perm_counts != expected_perms]
    if len(incorrect_counts) > 0:
        print(f"❌ Found {len(incorrect_counts)} questions with incorrect permutation counts:")
        print(incorrect_counts.head())
        return False
    else:
        print(f"✅ All questions have exactly {expected_perms} permutations")
    
    # Check permutation indices (should be 0-23)
    perm_indices = df['permutation_idx'].unique()
    expected_indices = set(range(24))
    if set(perm_indices) != expected_indices:
        print(f"❌ Permutation indices don't match expected range 0-23")
        print(f"Found: {sorted(perm_indices)}")
        return False
    else:
        print("✅ Permutation indices are correct (0-23)")
    
    # Check permutation strings
    perm_strings = df['permutation'].unique()
    print(f"✅ Found {len(perm_strings)} unique permutation strings")
    print(f"   Sample permutations: {sorted(perm_strings)[:5]}")
    
    return True

def verify_logits_and_predictions(df):
    """Verify that logits and predictions are consistent"""
    print(f"\n🧠 LOGITS & PREDICTIONS VERIFICATION")
    print("-" * 40)
    
    # Check that predicted_answer matches argmax of logits
    logit_cols = ['logit_A', 'logit_B', 'logit_C', 'logit_D']
    logits_array = df[logit_cols].values
    
    # Calculate argmax for each row
    computed_predictions = np.argmax(logits_array, axis=1)
    
    # Compare with stored predictions
    mismatches = df['predicted_answer'] != computed_predictions
    mismatch_count = mismatches.sum()
    
    if mismatch_count > 0:
        print(f"❌ Found {mismatch_count} prediction mismatches!")
        print("Sample mismatches:")
        mismatch_samples = df[mismatches][['question_id', 'predicted_answer'] + logit_cols].head()
        print(mismatch_samples)
        return False
    else:
        print("✅ All predictions match argmax of logits")
    
    # Check that is_correct is computed correctly
    correct_computed = (df['predicted_answer'] == df['correct_answer_position'])
    correct_mismatches = (df['is_correct'] != correct_computed).sum()
    
    if correct_mismatches > 0:
        print(f"❌ Found {correct_mismatches} is_correct computation errors!")
        return False
    else:
        print("✅ All is_correct values computed correctly")
    
    # Check logit ranges (should be reasonable)
    logit_stats = df[logit_cols].describe()
    print(f"\n📈 Logit statistics:")
    print(logit_stats.round(2))
    
    # Check for extreme values
    very_large = (np.abs(logits_array) > 100).any(axis=1).sum()
    if very_large > 0:
        print(f"⚠️  Found {very_large} rows with very large logits (>100)")
    
    return True

def verify_accuracy_consistency(df):
    """Verify that accuracy is consistent across permutations"""
    print(f"\n🎯 ACCURACY CONSISTENCY VERIFICATION")
    print("-" * 40)
    
    # Group by question and check if the same answer is predicted consistently
    question_groups = df.groupby('question_id')
    
    consistency_issues = 0
    for question_id, group in question_groups:
        # For the same question, the correct answer position should vary with permutation
        # but the model's confidence pattern should be somewhat consistent
        
        correct_positions = group['correct_answer_position'].unique()
        predictions = group['predicted_answer'].values
        is_correct = group['is_correct'].values
        
        # The correct answer should appear in different positions
        if len(correct_positions) == 1:
            consistency_issues += 1
    
    if consistency_issues > 0:
        print(f"⚠️  Found {consistency_issues} questions where correct answer doesn't vary across permutations")
    
    # Overall accuracy
    total_accuracy = df['is_correct'].mean() * 100
    print(f"📊 Overall accuracy: {total_accuracy:.2f}%")
    
    # Accuracy by subject
    subject_accuracy = df.groupby('subject')['is_correct'].mean().sort_values(ascending=False)
    print(f"\n🏆 Top 5 subjects by accuracy:")
    for subject, acc in subject_accuracy.head().items():
        print(f"   {subject}: {acc*100:.1f}%")
    
    print(f"\n📉 Bottom 5 subjects by accuracy:")
    for subject, acc in subject_accuracy.tail().items():
        print(f"   {subject}: {acc*100:.1f}%")
    
    return True

def verify_position_bias(df):
    """Check for position bias in predictions"""
    print(f"\n📍 POSITION BIAS VERIFICATION")
    print("-" * 40)
    
    # Count predictions by position
    position_counts = df['predicted_answer'].value_counts().sort_index()
    total_predictions = len(df)
    
    print("Prediction frequency by position:")
    for pos, count in position_counts.items():
        percentage = count / total_predictions * 100
        print(f"   Position {pos} ('{chr(65+pos)}'): {count:,} ({percentage:.1f}%)")
    
    # Expected is 25% for each position if no bias
    expected_pct = 25.0
    position_bias = {}
    for pos, count in position_counts.items():
        actual_pct = count / total_predictions * 100
        bias = actual_pct - expected_pct
        position_bias[pos] = bias
        
    max_bias = max(abs(bias) for bias in position_bias.values())
    print(f"\n📊 Maximum position bias: {max_bias:.1f} percentage points")
    
    if max_bias > 10:
        print("⚠️  Significant position bias detected!")
        for pos, bias in position_bias.items():
            if abs(bias) > 5:
                print(f"   Position {pos}: {bias:+.1f}pp bias")
    else:
        print("✅ Position bias is within reasonable range")
    
    return position_bias

def create_verification_plots(df, filename):
    """Create verification plots"""
    print(f"\n📊 CREATING VERIFICATION PLOTS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Verification Plots: {filename}', fontsize=16)
    
    # 1. Accuracy by subject
    subject_acc = df.groupby('subject')['is_correct'].mean().sort_values(ascending=False)
    top_subjects = subject_acc.head(10)
    
    axes[0,0].barh(range(len(top_subjects)), top_subjects.values)
    axes[0,0].set_yticks(range(len(top_subjects)))
    axes[0,0].set_yticklabels([s.replace('_', ' ').title()[:20] for s in top_subjects.index])
    axes[0,0].set_xlabel('Accuracy')
    axes[0,0].set_title('Top 10 Subjects by Accuracy')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # 2. Position bias
    position_counts = df['predicted_answer'].value_counts().sort_index()
    positions = [f'Position {i} ({chr(65+i)})' for i in position_counts.index]
    
    axes[0,1].bar(positions, position_counts.values)
    axes[0,1].set_title('Prediction Frequency by Position')
    axes[0,1].set_ylabel('Count')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Add expected line
    expected = len(df) / 4
    axes[0,1].axhline(y=expected, color='red', linestyle='--', label=f'Expected ({expected:.0f})')
    axes[0,1].legend()
    
    # 3. Logit distribution
    logit_cols = ['logit_A', 'logit_B', 'logit_C', 'logit_D']
    logits_flat = df[logit_cols].values.flatten()
    
    axes[1,0].hist(logits_flat, bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Logit Value')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Logit Values')
    axes[1,0].grid(alpha=0.3)
    
    # 4. Accuracy by permutation
    perm_acc = df.groupby('permutation_idx')['is_correct'].mean()
    
    axes[1,1].plot(perm_acc.index, perm_acc.values, 'o-')
    axes[1,1].set_xlabel('Permutation Index')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_title('Accuracy by Permutation')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    # plot_filename = filename.replace('.csv', '_verification.png')
    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    # print(f"✅ Saved verification plots to: {plot_filename}")
    plt.close()

def main():
    """Main verification function"""
    print("🔍 MMLU Forward Dataset Verification")
    print("="*60)
    
    # Find all forward pass files
    files = find_forward_files()
    if not files:
        return
    
    all_passed = True
    
    for filename in files:
        try:
            # Load the dataset
            print(f"\n📂 Loading {filename}...")
            df = pd.read_csv(filename)
            
            # Run all verifications
            checks = [
                verify_file_structure(df, filename),
                verify_permutations(df),
                verify_logits_and_predictions(df),
                verify_accuracy_consistency(df),
            ]
            
            # Position bias check (doesn't affect pass/fail)
            verify_position_bias(df)
            
            # Create plots
            try:
                create_verification_plots(df, filename)
            except Exception as e:
                print(f"⚠️  Could not create plots: {e}")
            
            # Overall result for this file
            file_passed = all(checks)
            if file_passed:
                print(f"\n✅ {filename}: ALL CHECKS PASSED")
            else:
                print(f"\n❌ {filename}: SOME CHECKS FAILED")
                all_passed = False
                
        except Exception as e:
            print(f"\n💥 Error processing {filename}: {e}")
            all_passed = False
    
    # Final summary
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL FILES PASSED VERIFICATION!")
        print("Your forward pass dataset is ready for calibration analysis.")
    else:
        print("⚠️  SOME FILES FAILED VERIFICATION!")
        print("Please check the errors above and re-run the forward pass script.")
    print("="*60)

if __name__ == "__main__":
    main()
