#!/usr/bin/env bash
# Training-time + VRAM logger for baseline wrappers, so baselines record training
# efficiency (total wall time, peak GPU memory, GPU name) like our arch runs do.
# Usage: source scripts/lib_timer.sh ; timer_start ; <train> ; timer_stop "$OUT/train_efficiency.json"

timer_start() {
  _TIMER_T0=$(date +%s)
  _TIMER_VRAM_FILE=$(mktemp 2>/dev/null || echo "/tmp/_timer_vram.$$")
  : > "$_TIMER_VRAM_FILE"
  if command -v nvidia-smi >/dev/null 2>&1; then
    ( while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
        sleep 15
      done ) >> "$_TIMER_VRAM_FILE" 2>/dev/null &
    _TIMER_SAMPLER=$!
  else
    _TIMER_SAMPLER=""
  fi
}

timer_stop() {
  local out="${1:-train_efficiency.json}"
  local dur=$(( $(date +%s) - ${_TIMER_T0:-$(date +%s)} ))
  [[ -n "${_TIMER_SAMPLER:-}" ]] && kill "$_TIMER_SAMPLER" 2>/dev/null || true
  local peak=0
  [[ -s "${_TIMER_VRAM_FILE:-/nonexistent}" ]] && peak=$(sort -n "$_TIMER_VRAM_FILE" 2>/dev/null | tail -1)
  local gpu="unknown"
  command -v nvidia-smi >/dev/null 2>&1 && gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  mkdir -p "$(dirname "$out")" 2>/dev/null || true
  printf '{"total_time_s": %s, "peak_vram_mib": %s, "gpu": "%s"}\n' "${dur:-0}" "${peak:-0}" "${gpu:-unknown}" > "$out"
  rm -f "$_TIMER_VRAM_FILE" 2>/dev/null || true
  echo "[timer] total_time_s=${dur} peak_vram_mib=${peak} gpu=${gpu} -> $out"
}
