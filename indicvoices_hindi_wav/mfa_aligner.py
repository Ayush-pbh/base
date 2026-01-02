import os
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import re

def text_to_phonemes(text, language='hi'):
    """
    Convert text to phonemes using espeak-ng
    
    Args:
        text: Input text in Devanagari
        language: Language code (hi for Hindi)
    
    Returns:
        Phoneme string in IPA format
    """
    try:
        # Run espeak-ng with IPA output
        result = subprocess.run(
            ['espeak-ng', '-v', language, '-q', '--ipa', '-x', text],
            capture_output=True,
            text=True,
            check=True
        )
        
        phonemes = result.stdout.strip()
        
        # Clean up the output
        # Remove excessive spaces
        phonemes = re.sub(r'\s+', ' ', phonemes)
        # Remove trailing/leading spaces
        phonemes = phonemes.strip()
        
        return phonemes
    
    except subprocess.CalledProcessError as e:
        print(f"Error processing text: {text[:50]}...")
        print(f"Error: {e}")
        return ""
    except FileNotFoundError:
        print("ERROR: espeak-ng not found! Please install it first.")
        print("Run: sudo apt-get install espeak-ng")
        exit(1)

def validate_phonemes(phonemes):
    """Check if phonemes look valid"""
    if not phonemes or len(phonemes) == 0:
        return False
    # Should contain IPA characters
    if len(phonemes) < 2:
        return False
    return True

def fix_audio_path(wav_path, speaker_dir):
    """
    Fix .flac to .wav mismatch and verify file exists
    
    Args:
        wav_path: Path from metadata (e.g., wavs/000000.flac)
        speaker_dir: Speaker directory path
    
    Returns:
        Corrected path or None if file doesn't exist
    """
    # Try original path first
    full_path = os.path.join(speaker_dir, wav_path)
    if os.path.exists(full_path):
        return wav_path
    
    # Try replacing .flac with .wav
    if wav_path.endswith('.flac'):
        wav_corrected = wav_path.replace('.flac', '.wav')
        full_path_corrected = os.path.join(speaker_dir, wav_corrected)
        if os.path.exists(full_path_corrected):
            return wav_corrected
    
    # Try replacing .wav with .flac (in case it's opposite)
    if wav_path.endswith('.wav'):
        flac_corrected = wav_path.replace('.wav', '.flac')
        full_path_corrected = os.path.join(speaker_dir, flac_corrected)
        if os.path.exists(full_path_corrected):
            return flac_corrected
    
    return None

def process_speaker_directory(speaker_dir):
    """
    Process all files for one speaker
    
    Args:
        speaker_dir: Path to speaker directory (e.g., S4256602600322595)
    
    Returns:
        DataFrame with phonemes added and paths corrected
    """
    metadata_path = os.path.join(speaker_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"⚠️  No metadata.csv found in {speaker_dir}")
        return None
    
    # Read metadata
    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"❌ Error reading {metadata_path}: {e}")
        return None
    
    # Check required columns
    required_cols = ['wav', 'text', 'speaker']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing columns in {metadata_path}")
        print(f"   Expected: {required_cols}")
        print(f"   Found: {list(df.columns)}")
        return None
    
    speaker_name = os.path.basename(speaker_dir)
    print(f"\n📁 Processing {speaker_name}")
    print(f"   Total samples: {len(df)}")
    
    # Fix audio paths and add phonemes
    phonemes_list = []
    corrected_paths = []
    failed_count = 0
    missing_audio = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Phonemizing"):
        text = row['text']
        wav_path = row['wav']
        
        # Fix audio path (.flac -> .wav)
        corrected_path = fix_audio_path(wav_path, speaker_dir)
        
        if corrected_path is None:
            print(f"   ⚠️  Row {idx}: Audio file not found: {wav_path}")
            corrected_paths.append(wav_path)  # Keep original
            phonemes_list.append('')
            missing_audio += 1
            failed_count += 1
            continue
        
        corrected_paths.append(corrected_path)
        
        # Check text
        if pd.isna(text) or text.strip() == '':
            print(f"   ⚠️  Row {idx}: Empty text")
            phonemes_list.append('')
            failed_count += 1
            continue
        
        # Get phonemes
        phonemes = text_to_phonemes(text)
        
        # Validate
        if not validate_phonemes(phonemes):
            print(f"   ⚠️  Row {idx}: Invalid phonemes for text: {text[:50]}...")
            failed_count += 1
        
        phonemes_list.append(phonemes)
    
    # Update dataframe
    df['wav'] = corrected_paths  # Fix .flac -> .wav
    df['phonemes'] = phonemes_list
    
    # Report
    print(f"   ✅ Completed: {len(df) - failed_count}/{len(df)} successful")
    if failed_count > 0:
        print(f"   ⚠️  Failed: {failed_count} samples")
    if missing_audio > 0:
        print(f"   ⚠️  Missing audio files: {missing_audio}")
    
    return df

def main(base_dir='.', output_dir='phonemized_data'):
    """
    Main function to process all speaker directories
    
    Args:
        base_dir: Base directory containing speaker folders
        output_dir: Directory to save phonemized metadata
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all speaker directories (S* pattern for IndicVoices)
    speaker_dirs = sorted([d for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith('S') 
                          and d.name not in ['scripts', 'samples']])  # Exclude common non-speaker dirs
    
    if not speaker_dirs:
        print("❌ No speaker directories found (S*)")
        print(f"   Looking in: {base_path.absolute()}")
        return
    
    print(f"🎤 Found {len(speaker_dirs)} speaker(s)")
    print(f"📤 Output will be saved to: {output_path.absolute()}\n")
    
    # Test espeak-ng first
    print("🧪 Testing espeak-ng...")
    test_text = "नमस्ते"
    test_phonemes = text_to_phonemes(test_text)
    print(f"   Test: '{test_text}' → '{test_phonemes}'")
    if not test_phonemes:
        print("❌ espeak-ng test failed!")
        return
    print("   ✅ espeak-ng working!\n")
    
    # Process each speaker
    all_data = []
    stats = {
        'total_speakers': len(speaker_dirs),
        'processed_speakers': 0,
        'total_samples': 0,
        'successful_samples': 0,
        'failed_samples': 0,
        'missing_audio': 0
    }
    
    for speaker_dir in speaker_dirs:
        df = process_speaker_directory(speaker_dir)
        
        if df is not None:
            # Save individual speaker phonemized data
            speaker_output = output_path / speaker_dir.name
            speaker_output.mkdir(exist_ok=True)
            
            output_csv = speaker_output / 'metadata.csv'
            df.to_csv(output_csv, index=False)
            print(f"   💾 Saved to: {output_csv}")
            
            # Collect stats
            stats['processed_speakers'] += 1
            stats['total_samples'] += len(df)
            
            valid_phonemes = df['phonemes'].apply(validate_phonemes).sum()
            stats['successful_samples'] += valid_phonemes
            stats['failed_samples'] += len(df) - valid_phonemes
            
            all_data.append(df)
    
    # Create combined dataset
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = output_path / 'combined_metadata.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\n📊 Combined dataset saved to: {combined_path}")
        
        # Show some sample phonemes
        print("\n🔍 Sample phonemized entries:")
        print("="*80)
        for i in range(min(3, len(combined_df))):
            row = combined_df.iloc[i]
            print(f"Text:     {row['text'][:60]}...")
            print(f"Phonemes: {row['phonemes'][:70]}...")
            print("-"*80)
    
    # Print summary
    print("\n" + "="*50)
    print("📈 SUMMARY")
    print("="*50)
    print(f"Speakers processed: {stats['processed_speakers']}/{stats['total_speakers']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful: {stats['successful_samples']}")
    print(f"Failed: {stats['failed_samples']}")
    if stats['total_samples'] > 0:
        print(f"Success rate: {(stats['successful_samples']/stats['total_samples'])*100:.1f}%")
    print("="*50)
    
    # Calculate total duration (approximate)
    total_hours = stats['total_samples'] * 3 / 3600  # Assuming avg 3 sec per sample
    print(f"\n⏱️  Estimated total audio: ~{total_hours:.1f} hours")
    print("="*50)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phonemize Hindi dataset using espeak-ng')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory containing speaker folders (default: current directory)')
    parser.add_argument('--output_dir', type=str, default='phonemized_data',
                        help='Output directory for phonemized data (default: phonemized_data)')
    
    args = parser.parse_args()
    
    main(args.base_dir, args.output_dir)