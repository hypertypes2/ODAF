#!/usr/bin/env bash
set -euo pipefail

DDK_ROOT="${1:-$HOME/lne_ddk-lneddk_3.0-2609}"
WRAPPER_DIR="$HOME/lne_toolchain_wrapper"

log() {
  printf '[setup_lne_vm] %s\n' "$*"
}

fail() {
  printf '[setup_lne_vm][FAIL] %s\n' "$*" >&2
  exit 1
}

log "DDK root: $DDK_ROOT"

if [ ! -d "$DDK_ROOT" ]; then
  fail "DDK directory not found: $DDK_ROOT"
fi

required_scripts=(
  "$DDK_ROOT/compile.sh"
  "$DDK_ROOT/run_spike.sh"
  "$DDK_ROOT/model_verifier.sh"
)

for script in "${required_scripts[@]}"; do
  if [ ! -f "$script" ]; then
    fail "Required script missing: $script"
  fi
  chmod +x "$script" || fail "Failed to restore execute permission for $script"
  log "Restored exec bit: $script"
done

if [ -d "$DDK_ROOT/install/bin" ]; then
  find "$DDK_ROOT/install/bin" -type f -exec chmod +x {} + || fail "Failed to restore install/bin permissions"
  log "Restored install/bin permissions"
else
  log "WARN: $DDK_ROOT/install/bin not found"
fi

if [ -d "$DDK_ROOT/toolchain/bin" ]; then
  find "$DDK_ROOT/toolchain/bin" -type f -exec chmod +x {} + || fail "Failed to restore toolchain/bin permissions"
  log "Restored toolchain/bin permissions"
else
  log "WARN: $DDK_ROOT/toolchain/bin not found"
fi

if [ -d "$DDK_ROOT/toolchain/libexec" ]; then
  find "$DDK_ROOT/toolchain/libexec" -type f -exec chmod +x {} + || fail "Failed to restore toolchain/libexec permissions"
  log "Restored toolchain/libexec permissions"
else
  log "WARN: $DDK_ROOT/toolchain/libexec not found"
fi

mkdir -p "$WRAPPER_DIR" || fail "Failed to create wrapper directory: $WRAPPER_DIR"

as_target="$DDK_ROOT/toolchain/bin/riscv32-unknown-elf-as"
ld_target="$DDK_ROOT/toolchain/bin/riscv32-unknown-elf-ld"

for target in "$as_target" "$ld_target"; do
  if [ ! -e "$target" ]; then
    fail "Required toolchain binary not found: $target"
  fi
done

if [ -L "$WRAPPER_DIR/as" ] || [ -f "$WRAPPER_DIR/as" ]; then
  rm -f "$WRAPPER_DIR/as" || fail "Failed to remove existing wrapper: $WRAPPER_DIR/as"
fi
if [ -L "$WRAPPER_DIR/ld" ] || [ -f "$WRAPPER_DIR/ld" ]; then
  rm -f "$WRAPPER_DIR/ld" || fail "Failed to remove existing wrapper: $WRAPPER_DIR/ld"
fi

ln -s "$as_target" "$WRAPPER_DIR/as" || fail "Failed to create as symlink"
ln -s "$ld_target" "$WRAPPER_DIR/ld" || fail "Failed to create ld symlink"

log "Wrapper configuration complete"
log "Current symlinks:"
ls -l "$WRAPPER_DIR"

printf '\n[PASS] LNE VM setup completed\n'
