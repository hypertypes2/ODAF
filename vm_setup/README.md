# LNE Worker VM Setup

This folder contains the minimal VM reproduction instructions and scripts needed to prepare the Linux worker used by ODAF Stage 3.

Important:
- The LNE DDK source tree and toolchain binaries are not stored in this Git repository.
- The repository contains only the setup and verification instructions needed to recreate the expected environment on a Linux VM worker.
- Secret values, API keys, SSH private keys, and runtime outputs are never committed to Git.

## 1. VM role

The Windows ODAF app orchestrates the Stage 3 workflow by connecting to a remote Linux worker over SSH/SCP.

The Linux worker is responsible for:
- receiving the generated TFLite model and raw calibration inputs
- preparing the DDK workspace
- running LNE compile
- running SPIKE validation
- generating summary and log outputs for ODAF to parse

## 2. Required DDK paths

The verified VM path in the current environment is:

- Worker user: `worker`
- DDK root: `~/lne_ddk-lneddk_3.0-2609`
- Toolchain wrapper: `~/lne_toolchain_wrapper`
- Target: `dqc`

Examples:

```bash
cd ~/lne_ddk-lneddk_3.0-2609
./compile.sh -t dqc
./run_spike.sh --net ./lne.json
```

## 3. DDK is not part of the repository

The DDK source tree, toolchain binary tree, and generated artifacts are runtime resources for the Linux worker and must not be copied into this Git repository.

Do not commit:
- `lne_ddk-lneddk_3.0-2609/`
- `toolchain/`
- `install/`
- generated Linux VM output directories
- raw calibration data
- real build artifacts

Repository-only content should be the setup/verification scripts and documentation.

## 4. Fixing executable permissions after copying from Windows

When a DDK tree is copied from Windows or another platform, Linux `x` permissions may be lost. The following files commonly need permission restoration:

- `compile.sh`
- `run_spike.sh`
- `model_verifier.sh`
- `install/bin/*`
- `toolchain/bin/*`
- `toolchain/libexec/*`

Fix permissions with:

```bash
chmod +x ~/lne_ddk-lneddk_3.0-2609/compile.sh
chmod +x ~/lne_ddk-lneddk_3.0-2609/run_spike.sh
chmod +x ~/lne_ddk-lneddk_3.0-2609/model_verifier.sh
find ~/lne_ddk-lneddk_3.0-2609/install/bin -type f -exec chmod +x {} +
find ~/lne_ddk-lneddk_3.0-2609/toolchain/bin -type f -exec chmod +x {} +
find ~/lne_ddk-lneddk_3.0-2609/toolchain/libexec -type f -exec chmod +x {} +
```

Do not use `sudo` for repository-sensitive operations. Only restore execute bits in the DDK workspace for the worker user.

## 5. Toolchain wrapper setup

The toolchain wrapper is not a copied binary; it is a symbolic-link wrapper used to point to the actual toolchain binaries.

Create the wrapper directory:

```bash
mkdir -p ~/lne_toolchain_wrapper
```

Then create symbolic links like:

```bash
ln -sfn ~/lne_ddk-lneddk_3.0-2609/toolchain/bin/riscv32-unknown-elf-as ~/lne_toolchain_wrapper/as
ln -sfn ~/lne_ddk-lneddk_3.0-2609/toolchain/bin/riscv32-unknown-elf-ld ~/lne_toolchain_wrapper/ld
```

This ensures the wrapper points to the actual toolchain, not a copied binary blob.

## 6. LNE compile method

Use the verified DQC build command:

```bash
cd ~/lne_ddk-lneddk_3.0-2609
./compile.sh -t dqc
```

This is the supported build command used by the project.

Do not use:

```bash
./run_spike.sh temperature
```

That command is invalid for this environment and is intentionally excluded from the examples here.

## 7. SPIKE validation method

Use the verified validation command:

```bash
cd ~/lne_ddk-lneddk_3.0-2609
./run_spike.sh --net ./lne.json
```

This executes the network description file `lne.json` and generates summary outputs in the VM output folder for ODAF to parse.

## 8. How to check the environment is healthy

Check the required tools and paths:

```bash
ls -ld ~/lne_ddk-lneddk_3.0-2609
ls -l ~/lne_ddk-lneddk_3.0-2609/compile.sh
ls -l ~/lne_ddk-lneddk_3.0-2609/run_spike.sh
ls -l ~/lne_toolchain_wrapper
ls -l ~/lne_toolchain_wrapper/as
ls -l ~/lne_toolchain_wrapper/ld
file ~/lne_toolchain_wrapper/as
file ~/lne_toolchain_wrapper/ld
```

If the shell script files are not executable, restore permissions before running them.

## 9. Notes and warnings

- Windows copy operations may remove Linux executable permission bits.
- Symbolic links must point to the actual toolchain binaries, not a copied copy.
- Do not modify `compile.sh`, `run_spike.sh`, `model_verifier.sh`, or toolchain binaries in place.
- Keep all secret values in `.env` or the local environment only; never store them in Git.
- Generated logs and output artifacts are runtime material and should stay outside the Git repository.
- This repository contains only the instructions and scripts required to reproduce the VM setup, not the DDK payload itself.
