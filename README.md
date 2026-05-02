# Apnea ECG EEG Comparison

Implementation of a digital signal processing pipeline using machine learning to compare sleep apnea detection performance between ECG and EEG data on a multimodal dataset.

## Project structure

```text
Apnea-ECG-EEG-Comparison/
├── data/
│   └── mesa/              # NSRR/MESA dataset downloaded locally
├── README.md
├── LICENSE
└── .gitignore
```

The dataset must be downloaded locally inside:

```text
data/mesa
```

The raw dataset should not be uploaded to GitHub because it may be large and may require individual NSRR access approval.

## 1. Clone the repository

```bash
git clone https://github.com/gosema/Apnea-ECG-EEG-Comparison
cd Apnea-ECG-EEG-Comparison
```

Replace `<REPOSITORY_URL>` with the repository URL.

## 2. Create and activate the Python environment

Recommended using Conda:

```bash
conda create -n pds python=3.11 -y
conda activate pds
```

Install the Python dependencies used by the project. If the repository contains a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt` yet, install the basic scientific stack:

```bash
pip install numpy scipy pandas matplotlib scikit-learn jupyter ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name pds --display-name "Python (pds)"
```

## 3. Install Ruby and the NSRR gem

The MESA data are downloaded using the official NSRR gem. The latest NSRR gem requires Ruby 2.7.2 or newer.

### macOS

The default macOS Ruby may be too old. Install a newer Ruby with Homebrew:

```bash
brew install ruby
```

Check your machine architecture:

```bash
uname -m
```

If the result is `arm64`, add Homebrew Ruby to your PATH with:

```bash
echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

If the result is `x86_64`, use:

```bash
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Check that the active Ruby version is recent enough:

```bash
ruby -v
which ruby
```

Then install the NSRR gem:

```bash
gem install nsrr
```

### Linux

On Ubuntu or Debian, install Ruby and the required build tools:

```bash
sudo apt update
sudo apt install ruby-full build-essential zlib1g-dev -y
```

Check the Ruby version:

```bash
ruby -v
```

If the version is lower than 2.7.2, install a newer Ruby using a version manager such as `rbenv` or use your distribution's newer Ruby package.

Then install the NSRR gem:

```bash
gem install nsrr
```

If you get a permissions error, install the gem only for your user:

```bash
gem install --user-install nsrr
```

### Windows

Install Ruby using RubyInstaller:

1. Download RubyInstaller from <https://rubyinstaller.org/downloads/>
2. Install Ruby 3.x with the MSYS2 development toolchain option enabled.
3. Open a new PowerShell or Command Prompt.
4. Check the Ruby version:

```powershell
ruby -v
```

Then install the NSRR gem:

```powershell
gem install nsrr
```

## 4. Download the MESA dataset into `data/mesa`

From the repository root, create the data directory:

```bash
mkdir -p data/mesa
```

Download the MESA dataset:

```bash
nsrr download mesa --download-folder data/mesa
```

Introduce the NSRR token when you are asked. After the download, the structure should be:

```text
data/
└── mesa/
    ├── actigraphy/
    ├── datasets/
    ├── documentation/
    ├── forms/
    ├── overlap/
    └── polysomnography/
```

The exact contents may depend on the files available to your NSRR account and the approved data access level.

## 5. Prevent data and secrets from being committed

Make sure `.gitignore` contains at least:

```gitignore
data/mesa/
.env
*.token
```

Before committing, always check:

```bash
git status
```

The `data/mesa/` folder should not appear as files to be committed.

## 6. Verify the setup

Check that the NSRR command works:

```bash
nsrr --version
```

Check that the dataset folder exists:

```bash
ls data/mesa
```

On Windows PowerShell, use:

```powershell
Get-ChildItem data\mesa
```

Check that the Python environment works:

```bash
python --version
python -c "import numpy, scipy, pandas, sklearn; print('Environment OK')"
```

