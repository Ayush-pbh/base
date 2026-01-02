import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

def analyze_dataset(df):
    """Analyze dataset statistics"""
    print("\n📊 Dataset Analysis")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Total speakers: {df['speaker'].nunique()}")
    
    # Samples per speaker
    speaker_counts = df['speaker'].value_counts()
    print(f"\nSamples per speaker:")
    print(f"  Min: {speaker_counts.min()}")
    print(f"  Max: {speaker_counts.max()}")
    print(f"  Mean: {speaker_counts.mean():.1f}")
    print(f"  Median: {speaker_counts.median():.1f}")
    
    # Show top 5 speakers
    print(f"\nTop 5 speakers by sample count:")
    for speaker, count in speaker_counts.head(5).items():
        print(f"  {speaker}: {count} samples")
    
    return speaker_counts

def create_random_split(df, train_ratio=0.95, val_ratio=0.05, random_seed=42):
    """
    Random split - all speakers appear in both train and val
    Good for: Single-speaker-like training where we want style diversity
    """
    print(f"\n🎲 Creating random split ({train_ratio*100:.0f}% train, {val_ratio*100:.0f}% val)")
    
    # Stratify by speaker to maintain speaker distribution
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=df['speaker']  # Ensures each speaker is in both sets
    )
    
    print(f"\n✅ Split created:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    
    # Verify all speakers in both sets
    train_speakers = set(train_df['speaker'].unique())
    val_speakers = set(val_df['speaker'].unique())
    print(f"\n  Speakers in train: {len(train_speakers)}")
    print(f"  Speakers in val:   {len(val_speakers)}")
    print(f"  Overlap: {len(train_speakers & val_speakers)} (should be {df['speaker'].nunique()})")
    
    return train_df, val_df

def create_speaker_split(df, val_speakers=5, test_speakers=3, random_seed=42):
    """
    Speaker-level split - some speakers only in train, some only in val/test
    Good for: Zero-shot evaluation, testing generalization to unseen speakers
    """
    print(f"\n👥 Creating speaker-level split")
    print(f"   Val speakers: {val_speakers}, Test speakers: {test_speakers}")
    
    # Get all speakers sorted by sample count
    speaker_counts = df['speaker'].value_counts()
    all_speakers = speaker_counts.index.tolist()
    
    np.random.seed(random_seed)
    np.random.shuffle(all_speakers)
    
    # Allocate speakers
    test_speaker_list = all_speakers[:test_speakers]
    val_speaker_list = all_speakers[test_speakers:test_speakers+val_speakers]
    train_speaker_list = all_speakers[test_speakers+val_speakers:]
    
    # Create splits
    train_df = df[df['speaker'].isin(train_speaker_list)]
    val_df = df[df['speaker'].isin(val_speaker_list)]
    test_df = df[df['speaker'].isin(test_speaker_list)]
    
    print(f"\n✅ Split created:")
    print(f"  Train: {len(train_df)} samples from {len(train_speaker_list)} speakers")
    print(f"  Val:   {len(val_df)} samples from {len(val_speaker_list)} speakers")
    print(f"  Test:  {len(test_df)} samples from {len(test_speaker_list)} speakers")
    
    return train_df, val_df, test_df

def save_split(df, output_dir, split_name, base_audio_dir, copy_audio=False):
    """
    Save split with metadata and optionally copy audio files
    
    Args:
        df: DataFrame with the split
        output_dir: Output directory (e.g., 'dataset/train')
        split_name: Name of split ('train', 'val', 'test')
        base_audio_dir: Base directory containing speaker folders
        copy_audio: If True, copy audio files. If False, keep original paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving {split_name} split to {output_path}")
    
    if copy_audio:
        # Create audio directory
        audio_dir = output_path / 'audio'
        audio_dir.mkdir(exist_ok=True)
        
        # Copy audio files and update paths
        new_rows = []
        print(f"   Copying audio files...")
        
        for idx, row in df.iterrows():
            speaker = row['speaker']
            wav_path = row['wav']
            
            # Source audio file
            src_audio = Path(base_audio_dir) / speaker / wav_path
            
            if not src_audio.exists():
                print(f"   ⚠️  Audio not found: {src_audio}")
                continue
            
            # New filename: speaker_filename.wav
            new_filename = f"{speaker}_{Path(wav_path).name}"
            dst_audio = audio_dir / new_filename
            
            # Copy file
            shutil.copy2(src_audio, dst_audio)
            
            # Update row
            row_copy = row.copy()
            row_copy['wav'] = f'audio/{new_filename}'
            new_rows.append(row_copy)
        
        new_df = pd.DataFrame(new_rows)
        print(f"   ✅ Copied {len(new_df)} audio files")
    else:
        # Keep original paths (relative to speaker directories)
        new_df = df.copy()
        print(f"   📍 Using original audio paths")
    
    # Save metadata
    metadata_path = output_path / 'metadata.csv'
    new_df.to_csv(metadata_path, index=False)
    print(f"   ✅ Saved metadata: {metadata_path}")
    
    # Save speaker list
    speakers = new_df['speaker'].unique().tolist()
    speaker_list_path = output_path / 'speakers.txt'
    with open(speaker_list_path, 'w') as f:
        f.write('\n'.join(speakers))
    print(f"   ✅ Saved speaker list: {speaker_list_path}")
    
    # Save statistics
    stats = {
        'total_samples': len(new_df),
        'total_speakers': len(speakers),
        'samples_per_speaker': new_df['speaker'].value_counts().to_dict()
    }
    stats_path = output_path / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ Saved statistics: {stats_path}")
    
    return new_df

def main():
    """Main function to create splits"""
    
    # Configuration
    COMBINED_METADATA = 'phonemized_data/combined_metadata.csv'
    BASE_AUDIO_DIR = '.'  # Root directory containing speaker folders
    OUTPUT_DIR = 'dataset'
    SPLIT_TYPE = 'random'  # Options: 'random' or 'speaker'
    COPY_AUDIO = False  # Set to True to copy audio, False to keep original paths
    
    # Random split parameters
    TRAIN_RATIO = 0.95
    VAL_RATIO = 0.05
    
    # Speaker split parameters
    VAL_SPEAKERS = 5
    TEST_SPEAKERS = 3
    
    RANDOM_SEED = 42
    
    print("="*60)
    print("🎯 StyleTTS 2 - Dataset Splitting")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading data from: {COMBINED_METADATA}")
    df = pd.read_csv(COMBINED_METADATA)
    
    # Analyze
    speaker_counts = analyze_dataset(df)
    
    # Create splits
    if SPLIT_TYPE == 'random':
        print(f"\n🎲 Using RANDOM split strategy")
        train_df, val_df = create_random_split(
            df, 
            train_ratio=TRAIN_RATIO, 
            val_ratio=VAL_RATIO,
            random_seed=RANDOM_SEED
        )
        test_df = None
    
    elif SPLIT_TYPE == 'speaker':
        print(f"\n👥 Using SPEAKER-LEVEL split strategy")
        train_df, val_df, test_df = create_speaker_split(
            df,
            val_speakers=VAL_SPEAKERS,
            test_speakers=TEST_SPEAKERS,
            random_seed=RANDOM_SEED
        )
    
    else:
        raise ValueError(f"Unknown split type: {SPLIT_TYPE}")
    
    # Save splits
    print("\n" + "="*60)
    print("💾 Saving splits")
    print("="*60)
    
    save_split(train_df, f'{OUTPUT_DIR}/train', 'train', BASE_AUDIO_DIR, COPY_AUDIO)
    save_split(val_df, f'{OUTPUT_DIR}/val', 'val', BASE_AUDIO_DIR, COPY_AUDIO)
    
    if test_df is not None:
        save_split(test_df, f'{OUTPUT_DIR}/test', 'test', BASE_AUDIO_DIR, COPY_AUDIO)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ SPLIT COMPLETE")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Split type: {SPLIT_TYPE}")
    print(f"Audio copied: {COPY_AUDIO}")
    print("\nNext steps:")
    print("  1. Verify splits: python validate_splits.py")
    print("  2. Run forced alignment (MFA)")
    print("  3. Start training!")
    print("="*60)

if __name__ == '__main__':
    main()