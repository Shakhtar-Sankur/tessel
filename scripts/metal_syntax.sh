#!/usr/bin/env bash
# Type-checks every kernel tessel generates for Metal with clang, against a
# stand-in for Metal's standard library (scripts/metal_stub): a check that
# runs on Linux. The Mac CI job compiles them with Apple's own compiler.
#
# usage: scripts/metal_syntax.sh DIR   (DIR holding */kernel.metal, as
#        `cargo run --example metal_cases -- write DIR` leaves it)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=$1
n=0
for f in "$dir"/*/kernel.metal; do
  # Clang, unlike Metal, has no threadgroup variables local to a function;
  # declared static they are the same variable, once per threadgroup.
  sed 's/^  threadgroup /  static threadgroup /' "$f" > "$f.cc"
  clang++ -std=c++17 -fsyntax-only -Wall -Wno-unknown-attributes -Wno-unused-function -Wno-unused-variable \
    -Wno-unused-but-set-variable \
    -Werror -isystem "$here/metal_stub" -x c++ "$f.cc"
  rm "$f.cc"
  n=$((n + 1))
done
echo "$n kernels type-check"
