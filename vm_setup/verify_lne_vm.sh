#!/usr/bin/env bash
set -u

DDK_ROOT="${ODAF_LNE_DDK_ROOT:-$HOME/lne_ddk-lneddk_3.0-2609}"
WRAPPER_DIR="$HOME/lne_toolchain_wrapper"

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; }

check_path() {
  local p="$1"
  if [ -e "$p" ]; then
    pass "$p exists"
  else
    fail "$p missing"
  fi
}

check_exec() {
  local p="$1"
  if [ -x "$p" ]; then
    pass "$p executable"
  else
    fail "$p not executable"
  fi
}

check_link() {
  local p="$1"
  if [ -L "$p" ]; then
    pass "$p is a symbolic link"
    local target
    target="$(readlink "$p" || true)"
    if [ -n "$target" ]; then
      pass "link target: $target"
    else
      fail "$p link target unreadable"
    fi
  else
    fail "$p is not a symlink"
  fi
}

printf '== LNE VM verification ==\n'
if [ -d "$DDK_ROOT" ]; then
  pass "DDK root exists: $DDK_ROOT"
else
  fail "DDK root missing: $DDK_ROOT"
fi

for script in \
  "$DDK_ROOT/compile.sh" \
  "$DDK_ROOT/run_spike.sh" \
  "$DDK_ROOT/model_verifier.sh"; do
  if [ -f "$script" ]; then
    pass "$script present"
    check_exec "$script"
  else
    fail "$script missing"
  fi
done

for tool in \
  "riscv32-unknown-elf-gcc" \
  "riscv32-unknown-elf-as" \
  "riscv32-unknown-elf-ld"; do
  if command -v "$tool" >/dev/null 2>&1; then
    pass "$tool found in PATH"
  else
    fail "$tool missing in PATH"
  fi
done

check_link "$WRAPPER_DIR/as"
check_link "$WRAPPER_DIR/ld"

for candidate in \
  "$DDK_ROOT/install/bin/lne_mapper_22" \
  "$DDK_ROOT/install/bin/tf_converter_22"; do
  if [ -x "$candidate" ]; then
    pass "$candidate executable"
  else
    fail "$candidate missing or not executable"
  fi
  if command -v ldd >/dev/null 2>&1; then
    ldd "$candidate" 2>/dev/null || warn "ldd inspection unavailable for $candidate"
  fi
done

if [ -f "$DDK_ROOT/lne.json" ]; then
  pass "lne.json exists"
  python - <<'PY' "$DDK_ROOT/lne.json"
import json, sys
p = sys.argv[1]
try:
    with open(p, 'r', encoding='utf-8') as f:
        json.load(f)
    print('[PASS] lne.json valid JSON')
except Exception as exc:
    print(f'[FAIL] invalid JSON: {exc}', file=sys.stderr)
    raise SystemExit(1)
PY
else
  fail "lne.json missing"
fi

if [ -d "$DDK_ROOT/tflite" ]; then
  pass "tflite directory exists"
  find "$DDK_ROOT/tflite" -maxdepth 1 -type f -name '*.tflite' | sed 's#^#/#' || warn "no .tflite files in $DDK_ROOT/tflite"
else
  warn "tflite directory missing"
fi

if [ -d "$DDK_ROOT/raw_input" ]; then
  pass "raw_input directory exists"
  raw_count=$(find "$DDK_ROOT/raw_input" -maxdepth 1 -type f -name '*.bin' | wc -l | tr -d ' ')
  printf '[PASS] raw_input .bin files: %s\n' "$raw_count"
else
  warn "raw_input directory missing"
fi

if [ -d "$DDK_ROOT/output" ]; then
  pass "output directory exists"
else
  warn "output directory missing"
fi

printf '\n== verification summary ==\n'
