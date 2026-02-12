#!/usr/bin/env bash
set -euo pipefail

VENV_WANDB_BIN="${VENV_WANDB_BIN:-/Users/massimoraso/projects/aigp/.venv/bin/wandb}"
SSH_ALIAS="${SSH_ALIAS:-pc}"

log() {
  printf '%s\n' "$*" >&2
}

if [[ ! -x "${VENV_WANDB_BIN}" ]]; then
  log "wandb CLI not found at ${VENV_WANDB_BIN}"
  log "Install it first: /Users/massimoraso/projects/aigp/.venv/bin/pip install wandb"
  exit 1
fi

if "${VENV_WANDB_BIN}" login --verify >/dev/null 2>&1; then
  log "W&B auth already configured."
  exit 0
fi

extract_remote_key() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_ALIAS}" '
    $files = @(
      "$env:USERPROFILE\.netrc",
      "$env:USERPROFILE\_netrc",
      "$env:USERPROFILE\.config\wandb\settings",
      "$env:LOCALAPPDATA\wandb\settings",
      "$env:LOCALAPPDATA\wandb\wandb_settings"
    )
    foreach ($f in $files) {
      if (Test-Path $f) {
        $content = Get-Content -Raw -ErrorAction SilentlyContinue $f
        if ($null -eq $content) { continue }
        $m = [regex]::Match($content, "(wandb_v[0-9]+_[A-Za-z0-9_]{20,})")
        if ($m.Success) {
          Write-Output $m.Groups[1].Value
          exit 0
        }
        $m = [regex]::Match($content, "(?i)(?:password|api[_-]?key)\s*[:= ]\s*([A-Za-z0-9_]{40,})")
        if ($m.Success) {
          Write-Output $m.Groups[1].Value
          exit 0
        }
        $m2 = [regex]::Match($content, "\b[A-Za-z0-9_]{40,}\b")
        if ($m2.Success) {
          Write-Output $m2.Value
          exit 0
        }
      }
    }
    exit 1
  ' 2>/dev/null | tr -d '\r\n'
}

REMOTE_KEY=""
if REMOTE_KEY="$(extract_remote_key)" && [[ -n "${REMOTE_KEY}" ]]; then
  if "${VENV_WANDB_BIN}" login --relogin --verify "${REMOTE_KEY}" >/dev/null 2>&1; then
    unset REMOTE_KEY
    if "${VENV_WANDB_BIN}" login --verify >/dev/null 2>&1; then
      log "W&B auth configured from remote ${SSH_ALIAS} files."
      exit 0
    fi
  fi
fi
unset REMOTE_KEY

log "Remote extraction failed or unusable. Enter W&B API key manually."
if [[ ! -t 0 ]]; then
  log "Non-interactive shell: cannot prompt for key. Set WANDB_API_KEY or run interactively."
  exit 1
fi
printf 'W&B API key: ' >&2
stty -echo
IFS= read -r MANUAL_KEY
stty echo
printf '\n' >&2

if [[ -z "${MANUAL_KEY}" ]]; then
  log "No key provided."
  exit 1
fi

if ! "${VENV_WANDB_BIN}" login --relogin --verify "${MANUAL_KEY}" >/dev/null 2>&1; then
  unset MANUAL_KEY
  log "wandb login failed."
  exit 1
fi
unset MANUAL_KEY

if "${VENV_WANDB_BIN}" login --verify >/dev/null 2>&1; then
  log "W&B auth configured successfully."
  exit 0
fi

log "W&B auth still invalid after login."
exit 1
