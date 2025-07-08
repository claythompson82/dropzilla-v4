# GPU Setup Guide

The following steps outline how to prepare a WSL2 Ubuntu 22.04 environment with CUDA 12.9 for running Dropzilla v4 with GPU acceleration.

1. **Install NVIDIA Drivers**
   - Ensure the latest Windows NVIDIA drivers are installed.
   - Enable the WSL2 feature for NVIDIA in Windows.

2. **Install CUDA Toolkit 12.9**
   - Inside the Ubuntu shell, add the CUDA repository:
     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
     sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
     sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list
     sudo apt update && sudo apt install -y cuda-toolkit-12-9
     ```
   - Reboot the WSL2 instance after installation.

3. **Create and Activate a Python Environment**
   ```bash
   python3 -m venv ~/.venvs/dz-gpu
   source ~/.venvs/dz-gpu/bin/activate
   ```

4. **Install Python Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify the GPU Build**
   Run the verification script:
   ```bash
   python scripts/verify_gpu.py
   ```
   A successful setup prints:
   `LightGBM built with CUDA ✅`

