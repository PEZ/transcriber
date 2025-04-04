import os
import torch
import sys
import subprocess
import urllib.request
import ssl
from safetensors import safe_open
from collections import OrderedDict
import time

def download_file(url, dest_path, max_retries=3):
    """Download a file from a URL to a local destination with progress indicator."""
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Create a directory for the file if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Set up SSL context (with verification disabled)
    context = ssl._create_unverified_context()

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "torch.hub/1.0"}
            )

            with urllib.request.urlopen(req, context=context) as response:
                # Get file size if available
                file_size = int(response.info().get("Content-Length", 0))

                # Set up progress reporting
                if has_tqdm and file_size > 0:
                    pbar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024)

                # Download the file
                with open(dest_path, 'wb') as out_file:
                    block_size = 8192
                    downloaded = 0

                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break

                        out_file.write(buffer)
                        downloaded += len(buffer)

                        # Update progress
                        if has_tqdm and file_size > 0:
                            pbar.update(len(buffer))
                        elif not has_tqdm and file_size > 0:
                            # Simple progress indicator
                            done = int(50 * downloaded / file_size)
                            sys.stdout.write("\r[%s%s] %d%%" % ('=' * done, ' ' * (50-done), int(100 * downloaded / file_size)))
                            sys.stdout.flush()

                if has_tqdm and file_size > 0:
                    pbar.close()
                elif not has_tqdm and file_size > 0:
                    sys.stdout.write("\n")

                print(f"Downloaded {url} to {dest_path}")
                return True

        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason} for url {url}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Failed to download after {max_retries} attempts")
                print("Please check if the URL is correct or try downloading manually")
                raise
        except urllib.error.URLError as e:
            print(f"URL Error: {e.reason} for url {url}")
            print("Check your internet connection")
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise

    return False

def convert_safetensors_to_pytorch(input_dir, output_path):
    """Convert KB Whisper safetensors model to PyTorch format with expected structure"""
    print("Loading safetensors model...")

    # Load model from safetensors
    state_dict = OrderedDict()
    for i in range(1, 3):
        file = f'{input_dir}/model-0000{i}-of-00002.safetensors'
        with safe_open(file, framework='pt', device='cpu') as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # Add expected 'dims' structure that convert-pt-to-ggml.py expects
    model_dims = {
        "n_mels": 80,
        "n_vocab": 51865,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": 32
    }

    # Create the checkpoint with required structure
    checkpoint = {
        "model_state_dict": state_dict,
        "dims": model_dims
    }

    # Save as PyTorch format with expected structure
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    print("Conversion completed")
    return output_path

def convert_to_ggml(pytorch_model, whisper_cpp_path, output_dir, precision="f16"):
    """Convert PyTorch model to GGML format using whisper.cpp script"""
    print(f"Converting PyTorch model to GGML format ({precision})...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        import whisper
        whisper_assets_path = os.path.join(os.path.dirname(whisper.__file__), "assets")
        print(f"Found Whisper assets at: {whisper_assets_path}")
    except ImportError:
        print("ERROR: Missing required 'whisper' module.")
        print("Please set up your environment with: pip install -r requirements.txt")
        sys.exit(1)

    # Get source mel_filters.npz file
    mel_filters_source = os.path.join(whisper_assets_path, "mel_filters.npz")

    # Create directory structure expected by convert-pt-to-ggml.py script
    # The script looks for mel_filters.npz in '{whisper_cpp_path}/whisper/assets/'
    expected_mel_filters_dir = os.path.join(whisper_cpp_path, "whisper", "assets")
    expected_mel_filters_path = os.path.join(expected_mel_filters_dir, "mel_filters.npz")
    os.makedirs(expected_mel_filters_dir, exist_ok=True)

    # Copy mel_filters.npz to the expected location
    import shutil
    if not os.path.exists(expected_mel_filters_path):
        shutil.copy2(mel_filters_source, expected_mel_filters_path)
        print(f"Copied mel_filters.npz to {expected_mel_filters_path}")

    # Define URLs for tokenizer files - Updated to correct URLs that exist
    vocab_url = "https://huggingface.co/openai/whisper-large-v2/resolve/main/vocab.json"
    merges_url = "https://huggingface.co/openai/whisper-large-v2/resolve/main/merges.txt"

    # Create local directories for assets and copy tokenizer files
    # The script might also look for these files in a multilingual directory
    local_tokenizer_dir = os.path.join(whisper_cpp_path, "whisper", "assets", "multilingual")
    os.makedirs(local_tokenizer_dir, exist_ok=True)

    # Ensure tokenizer files are available
    for file_name, url in [("vocab.json", vocab_url), ("merges.txt", merges_url)]:
        dest_file = os.path.join(local_tokenizer_dir, file_name)
        if not os.path.exists(dest_file):
            if not download_file(url, dest_file):
                print(f"Failed to download {file_name}. Please download it manually from {url} to {dest_file}")
                sys.exit(1)

    # Path to the conversion script
    convert_script = os.path.join(whisper_cpp_path, "models", "convert-pt-to-ggml.py")

    # Run the conversion script
    print("Running conversion script...")
    result = subprocess.run(
        [sys.executable, convert_script, pytorch_model, whisper_cpp_path, output_dir, precision],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("GGML conversion successful")
        print(result.stdout)
    else:
        print("GGML conversion failed")
        print(result.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert-kb-model.py <input_dir> <whisper_cpp_dir> <output_dir> [precision]")
        print("Example: python convert-kb-model.py /tmp/kb-whisper-large ../whisper.cpp ./models f16")
        sys.exit(1)

    input_dir = sys.argv[1]
    whisper_cpp_path = sys.argv[2]
    output_dir = sys.argv[3]
    precision = sys.argv[4] if len(sys.argv) > 4 else "f16"

    # Pipeline of transformations
    pytorch_model = convert_safetensors_to_pytorch(input_dir,
                                                 os.path.join("./tmp", "kb-whisper-large", "pytorch_model.bin"))
    convert_to_ggml(pytorch_model, whisper_cpp_path, output_dir, precision)