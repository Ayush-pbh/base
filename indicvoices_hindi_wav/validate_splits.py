import pandas as pd
import os
from pathlib import Path

def validate_split(split_dir, base_audio_dir='.'):
    """Validate a single split"""
    metadata_path = Path(split_dir) / 'metadata.csv'
    
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return False
    
    df = pd.read_csv(metadata_path)
    split_name = Path(split_dir).name
    
    print(f"\n{'='*60}")
    print(f"📊 Validating {split_name.upper()} split")
    print(f"{'='*60}")
    print(f"Samples: {len(df)}")
    print(f"Speakers: {df['speaker'].nunique()}")
    
    # Check required columns
    required_cols = ['wav', 'text', 'speaker', 'phonemes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    print(f"✅ All required columns present")
    
    # Check for empty values
    empty_text = df['text'].isna().sum()
    empty_phonemes = df['phonemes'].isna().sum()
    if empty_text > 0:
        print(f"⚠️  Empty text: {empty_text} samples")
    if empty_phonemes > 0:
        print(f"⚠️  Empty phonemes: {empty_phonemes} samples")
    
    # Verify audio files exist (sample 10)
    print(f"\n🔍 Checking audio files (sampling 10)...")
    missing_audio = 0
    sample_indices = range(0, min(len(df), 100), 10)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        speaker = row['speaker']
        wav_path = row['wav']
        
        # Construct full path
        if 'audio/' in wav_path:
            # Copied audio
            full_path = Path(split_dir) / wav_path
        else:
            # Original path
            full_path = Path(base_audio_dir) / speaker / wav_path
        
        if not full_path.exists():
            print(f"   ❌ Missing: {full_path}")
            missing_audio += 1
    
    if missing_audio == 0:
        print(f"   ✅ All sampled audio files exist")
    else:
        print(f"   ⚠️  Missing audio files: {missing_audio}")
    
    # Show sample
    print(f"\n📝 Sample entries:")
    for i in range(min(2, len(df))):
        row = df.iloc[i]
        print(f"\n  {i+1}.")
        print(f"    Speaker: {row['speaker']}")
        print(f"    Audio: {row['wav']}")
        print(f"    Text: {row['text'][:50]}...")
        print(f"    Phonemes: {row['phonemes'][:60]}...")
    
    return True

def main():
    """Validate all splits"""
    dataset_dir = 'dataset'
    
    print("="*60)
    print("🔍 DATASET VALIDATION")
    print("="*60)
    
    # Check splits
    splits = ['train', 'val', 'test']
    valid_splits = []
    
    for split in splits:
        split_path = Path(dataset_dir) / split
        if split_path.exists():
            if validate_split(split_path):
                valid_splits.append(split)
        else:
            print(f"\nℹ️  Split not found: {split}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ VALIDATION SUMMARY")
    print("="*60)
    print(f"Valid splits: {', '.join(valid_splits)}")
    
    # Load and compare
    if 'train' in valid_splits and 'val' in valid_splits:
        train_df = pd.read_csv(f'{dataset_dir}/train/metadata.csv')
        val_df = pd.read_csv(f'{dataset_dir}/val/metadata.csv')
        
        total = len(train_df) + len(val_df)
        print(f"\nTotal samples: {total}")
        print(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/total*100:.1f}%)")
        
        # Check for data leakage (same audio in train and val)
        train_wavs = set(train_df['wav'])
        val_wavs = set(val_df['wav'])
        overlap = train_wavs & val_wavs
        
        if overlap:
            print(f"\n⚠️  WARNING: {len(overlap)} audio files in both train and val!")
        else:
            print(f"\n✅ No data leakage detected")
    
    print("="*60)

if __name__ == '__main__':
    main()
