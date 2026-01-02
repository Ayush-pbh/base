# export AWS_ACCESS_KEY_ID=xxx
# export AWS_SECRET_ACCESS_KEY=xxx
# export AWS_DEFAULT_REGION=ap-south-1

# python upload_folder_to_s3.py \
#   /workspace/indicvoices_hindi \
#   training-data-storage-sage \
#   datasets/indicvoices_hindi

import subprocess
import sys

def upload_folder_fast(local_dir, bucket, s3_prefix=""):
    s3_path = f"s3://{bucket}/{s3_prefix}" if s3_prefix else f"s3://{bucket}"
    
    cmd = [
        "aws", "s3", "sync",
        local_dir,
        s3_path,
        "--no-progress"  # Remove this if you want progress
    ]
    
    print(f"Uploading {local_dir} to {s3_path}")
    result = subprocess.run(cmd, check=True)
    print("Upload complete!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python upload.py <local_folder> <bucket> [s3_prefix]")
        sys.exit(1)
    
    upload_folder_fast(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")