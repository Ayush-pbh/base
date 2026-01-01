#!/bin/bash

# StyleTTS2 RunPod Environment Setup Script
# This script sets up a complete environment for training StyleTTS2

set -e  # Exit on error

echo "=========================================="
echo "StyleTTS2 RunPod Environment Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Update system packages
print_status "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    espeak-ng \
    ranger \
    tmux \
    vim \
    htop \
    libsndfile1 \
    ffmpeg

# Install Miniconda if conda is not available
if ! command -v conda &> /dev/null; then
    print_status "Installing Miniconda to /workspace..."
    mkdir -p /workspace
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /workspace/miniconda3
    rm /tmp/miniconda.sh
    
    # Initialize conda
    eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    # Accept Terms of Service and Privacy Policy
    print_status "Accepting Conda Terms of Service and Privacy Policy..."
    conda config --set channel_priority flexible
    /workspace/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    /workspace/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    
    print_status "Miniconda installed successfully at /workspace/miniconda3"
else
    print_status "Conda is already installed"
    eval "$(conda shell.bash hook)"
fi

# Create conda environment with Python 3.10.19
ENV_NAME="styletts2"
print_status "Creating conda environment: $ENV_NAME with Python 3.10.19..."

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=3.10.19 -y

# Activate the environment
print_status "Activating conda environment: $ENV_NAME..."
conda activate $ENV_NAME

# Install Python packages
print_status "Installing phonemizer..."
pip install phonemizer

# Install PyTorch (CUDA 11.8 version - adjust if needed)
print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone StyleTTS2 repository (optional - uncomment if needed)
# print_status "Cloning StyleTTS2 repository..."
# cd $HOME
# if [ -d "StyleTTS2" ]; then
#     print_warning "StyleTTS2 directory already exists"
# else
#     git clone https://github.com/yl4579/StyleTTS2.git
#     cd StyleTTS2
#     print_status "Installing StyleTTS2 requirements..."
#     pip install -r requirements.txt
# fi

# Verify installations
print_status "Verifying installations..."
echo ""
echo "Python version:"
python --version
echo ""
echo "Conda environments:"
conda env list
echo ""
echo "espeak-ng version:"
espeak-ng --version
echo ""
echo "Checking phonemizer:"
python -c "import phonemizer; print(f'phonemizer version: {phonemizer.__version__}')" || print_error "phonemizer not installed correctly"
echo ""

# Configure tmux (optional)
print_status "Creating tmux configuration..."
cat > $HOME/.tmux.conf << 'EOF'
# Enable mouse support
set -g mouse on

# Set prefix to Ctrl-a
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# Split panes using | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# Reload config
bind r source-file ~/.tmux.conf

# Start windows and panes at 1, not 0
set -g base-index 1
setw -g pane-base-index 1
EOF

# Create activation script
print_status "Creating activation script..."
cat > $HOME/activate_styletts2.sh << EOF
#!/bin/bash
# Quick activation script for StyleTTS2 environment
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME
echo "StyleTTS2 environment activated!"
echo "Python: \$(which python)"
echo "Python version: \$(python --version)"
EOF
chmod +x $HOME/activate_styletts2.sh

# Print completion message
echo ""
echo "=========================================="
print_status "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source ~/activate_styletts2.sh"
echo ""
echo "Or manually:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Installed tools:"
echo "  - Conda environment: $ENV_NAME (Python 3.10.19)"
echo "  - espeak-ng (for phoneme conversion)"
echo "  - phonemizer (Python library)"
echo "  - ranger (terminal file manager - run with 'ranger')"
echo "  - tmux (terminal multiplexer - run with 'tmux')"
echo ""
echo "Useful commands:"
echo "  - Start tmux session: tmux"
echo "  - Open file manager: ranger"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo ""
print_warning "Note: You may need to restart your shell or run 'source ~/.bashrc' for conda to work properly"