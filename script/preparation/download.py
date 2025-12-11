import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
from huggingface_hub import snapshot_download

def download_with_retry(repo_id, repo_type, local_dir, max_retries=5, split=None, allow_patterns=None):
    """Download with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}...")
            snapshot_download(
                repo_id=repo_id, 
                repo_type=repo_type, 
                local_dir=local_dir, 
                allow_patterns=allow_patterns  # 启用文件过滤
            )
            print("Download completed successfully!")
            return
        except Exception as e:
            print(f"Download failed on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("All download attempts failed!")
                raise

# Execute download with retry
download_with_retry(
    repo_id="open-r1/OpenR1-Math-220k", 
    repo_type="dataset", 
    local_dir="data/raw_data", 
    # split="train",
    allow_patterns=["data/**"],
    max_retries=100
)
