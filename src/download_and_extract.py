import os
import urllib.request
import subprocess
from tqdm import tqdm
import shutil
import sys

# URLs and target filenames
files = {
    "https://zenodo.org/records/12665499/files/BrushlessMotor.7z?download=1": "BrushlessMotor.7z",
    "https://zenodo.org/records/12665499/files/RoboticArm.7z?download=1": "RoboticArm.7z"
}

data_dir = "data"
output_dir = "datasets"

def download_with_progress(url, dest_path, desc):
    """
    Download a file from a URL with a tqdm progress bar.
    """
    with tqdm(total=100, desc=desc, unit="%", ncols=100) as pbar:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                downloaded = blocknum * blocksize
                percent = int(downloaded * 100 / totalsize)
                pbar.update(percent - pbar.n)
        urllib.request.urlretrieve(url, dest_path, reporthook)

def download_and_extract():
    """
    Download and extract datasets from specified URLs.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Check for 7z executable
    if not shutil.which("7z"):
        print("❌ Error: '7z' is not found in PATH. Please install 7-Zip and make sure it's available in the terminal.")
        sys.exit(1)

    for url, filename in files.items():
        zip_path = os.path.join(data_dir, filename)
        dataset_name = os.path.splitext(filename)[0]
        extracted_path = os.path.join(output_dir, dataset_name)

        # 1. Download
        if not os.path.exists(zip_path):
            try:
                download_with_progress(url, zip_path, f"Downloading {filename}")
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
                continue
        else:
            print(f"✔️ Skipping download, already exists: {filename}")

        # 2. Extract
        if not os.path.exists(extracted_path):
            try:
                with tqdm(total=100, desc=f"Extracting {filename}", unit="%", ncols=100) as pbar:
                    result = subprocess.run(
                        ["7z", "x", zip_path, f"-o{output_dir}"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if result.returncode != 0:
                        print(f"❌ Extraction error for {filename}:\n{result.stderr}")
                    else:
                        pbar.update(100)
                        print(f"✔️ Extracted {filename} successfully.\n")
            except Exception as e:
                print(f"❌ Error extracting {filename}: {e}")
        else:
            print(f"✔️ Skipping extraction, folder already exists: {extracted_path}")

if __name__ == "__main__":
    download_and_extract()
