import os
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= CONFIG =================
SOURCE_DIR = "/workspace/indicvoices_hindi"
OUTPUT_DIR = "/workspace/indicvoices_hindi_wav"
SAMPLE_RATE = 24000
BIT_DEPTH = 16
MAX_WORKERS = 8  # Adjust based on CPU cores
# =========================================

def convert_flac_to_wav(flac_path, wav_path):
    """Convert a single FLAC to WAV using ffmpeg"""
    cmd = [
        "ffmpeg",
        "-i", flac_path,
        "-ar", str(SAMPLE_RATE),      # Sample rate 24kHz
        "-ac", "1",                     # Mono
        "-sample_fmt", "s16",           # 16-bit PCM
        "-y",                           # Overwrite
        wav_path
    ]
    
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    return flac_path

def process_speaker(speaker_dir, source_root, output_root):
    """Process all files for one speaker"""
    speaker_name = speaker_dir.name
    
    # Create output directories
    output_speaker_dir = output_root / speaker_name
    output_wavs_dir = output_speaker_dir / "wavs"
    output_wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy metadata.csv
    metadata_src = speaker_dir / "metadata.csv"
    metadata_dst = output_speaker_dir / "metadata.csv"
    if metadata_src.exists():
        shutil.copy2(metadata_src, metadata_dst)
    
    # Collect all FLAC files
    wavs_dir = speaker_dir / "wavs"
    flac_files = list(wavs_dir.glob("*.flac"))
    
    # Convert each FLAC to WAV
    tasks = []
    for flac_file in flac_files:
        wav_name = flac_file.stem + ".wav"
        wav_path = output_wavs_dir / wav_name
        tasks.append((str(flac_file), str(wav_path)))
    
    return tasks

def main():
    source_root = Path(SOURCE_DIR)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(exist_ok=True)
    
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Target: 24kHz, 16-bit, Mono WAV\n")
    
    # Collect all speaker directories
    speaker_dirs = [d for d in source_root.iterdir() if d.is_dir()]
    print(f"Found {len(speaker_dirs)} speakers")
    
    # Collect all conversion tasks
    all_tasks = []
    print("Scanning files...")
    for speaker_dir in tqdm(speaker_dirs, desc="Scanning speakers"):
        tasks = process_speaker(speaker_dir, source_root, output_root)
        all_tasks.extend(tasks)
    
    print(f"\nTotal files to convert: {len(all_tasks)}")
    
    # Convert all files in parallel
    print(f"Converting with {MAX_WORKERS} workers...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(convert_flac_to_wav, flac, wav)
            for flac, wav in all_tasks
        ]
        
        with tqdm(total=len(futures), desc="Converting") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    pbar.write(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"Output directory: {output_root}")

if __name__ == "__main__":
    main()