#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[Setup Acados] $*"
}

# Check if environment variable is already set
if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
  log "Not running inside a Pixi environment; skipping setup_acados.sh"
  exit 0
fi

# Check if pixi env is properly set up
if [ ! -f "${PIXI_PROJECT_ROOT}/pixi.lock" ]; then
  log "ERROR: pixi environment is not properly set up."
  exit 0
fi

ACADOS_DIR="${PIXI_PROJECT_ROOT}/acados"
TERA_RENDERER_VERSION="0.2.0"

case "$(uname -s)" in
  Linux)
    ACADOS_PLATFORM="linux"
    ACADOS_LIB_NAME="libacados.so"
    ;;
  Darwin)
    ACADOS_PLATFORM="osx"
    ACADOS_LIB_NAME="libacados.dylib"
    ;;
  *)
    log "ERROR: Unsupported platform: $(uname -s)"
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64 | amd64)
    ACADOS_ARCH="amd64"
    ;;
  arm64 | aarch64)
    ACADOS_ARCH="arm64"
    ;;
  *)
    log "ERROR: Unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac

ACADOS_LIB="${ACADOS_DIR}/lib/${ACADOS_LIB_NAME}"
TERA_RENDERER="${ACADOS_DIR}/bin/t_renderer"
TERA_RENDERER_URL="https://github.com/acados/tera_renderer/releases/download/v${TERA_RENDERER_VERSION}/t_renderer-v${TERA_RENDERER_VERSION}-${ACADOS_PLATFORM}-${ACADOS_ARCH}"

jobs_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    echo 1
  fi
}

should_download_tera_renderer() {
  if [ ! -f "$TERA_RENDERER" ] || [ ! -x "$TERA_RENDERER" ]; then
    return 0
  fi

  set +e
  "$TERA_RENDERER" --help >/dev/null 2>&1
  local status=$?
  set -e

  case "$status" in
    126 | 127)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ensure_macos_dylib_rpaths() {
  if [ "$ACADOS_PLATFORM" != "osx" ] || [ ! -d "${ACADOS_DIR}/lib" ]; then
    return
  fi

  local otool_cmd="${OTOOL:-otool}"
  local install_name_tool_cmd="${INSTALL_NAME_TOOL:-install_name_tool}"

  if ! command -v "$otool_cmd" >/dev/null 2>&1; then
    log "WARNING: otool not found; skipping macOS rpath setup."
    return
  fi

  if ! command -v "$install_name_tool_cmd" >/dev/null 2>&1; then
    log "WARNING: install_name_tool not found; skipping macOS rpath setup."
    return
  fi

  local dylib
  for dylib in "${ACADOS_DIR}"/lib/*.dylib; do
    [ -e "$dylib" ] || continue
    if ! "$otool_cmd" -l "$dylib" | grep -Fq "path ${ACADOS_DIR}/lib"; then
      "$install_name_tool_cmd" -add_rpath "${ACADOS_DIR}/lib" "$dylib"
    fi
  done
}

# Clone and build acados
if [ ! -d "${ACADOS_DIR}/.git" ]; then
  log "Cloning acados..."
  git clone https://github.com/acados/acados.git "${ACADOS_DIR}"
  (
    cd "${ACADOS_DIR}"
    git checkout tags/v0.5.1
    git submodule update --recursive --init
  )
fi

# Check if pip is installed
if ! command -v pip >/dev/null 2>&1; then
  log "ERROR: pip is not installed. Please install pip first."
  exit 0
fi

# Build Acados
if [ ! -f "$ACADOS_LIB" ]; then
  log "Building acados..."
  mkdir -p "${ACADOS_DIR}/build"
  cmake_args=(-DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
  if [ "$ACADOS_PLATFORM" = "osx" ] && [ "$ACADOS_ARCH" = "arm64" ]; then
    cmake_args+=(-DBLASFEO_TARGET=ARMV8A_APPLE_M1)
  fi
  if [ "$ACADOS_PLATFORM" = "osx" ]; then
    cmake_args+=(-DCMAKE_INSTALL_RPATH="${ACADOS_DIR}/lib" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON)
  fi
  (
    cd "${ACADOS_DIR}/build"
    cmake "${cmake_args[@]}" ..
    make install -j"$(jobs_count)"
  )
fi

ensure_macos_dylib_rpaths

# Install Acados Python interface
if ! pip show acados-template >/dev/null 2>&1; then
  log "Installing acados Python interface..."
  pip install -e "${ACADOS_DIR}/interfaces/acados_template"
fi

# Download Tera Renderer
if should_download_tera_renderer; then
  log "Downloading tera_renderer for ${ACADOS_PLATFORM}-${ACADOS_ARCH}..."
  mkdir -p "${ACADOS_DIR}/bin"
  curl -fL "$TERA_RENDERER_URL" -o "$TERA_RENDERER"
  chmod +x "$TERA_RENDERER"
fi

# Setting Environment Variables
if [ -f "$ACADOS_LIB" ]; then
  export ACADOS_SOURCE_DIR="$ACADOS_DIR"
  export ACADOS_INSTALL_DIR="$ACADOS_DIR"
  if [ "$ACADOS_PLATFORM" = "osx" ]; then
    export DYLD_LIBRARY_PATH="${ACADOS_DIR}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
    export LDFLAGS="-Wl,-rpath,${ACADOS_DIR}/lib${LDFLAGS:+ ${LDFLAGS}}"
  else
    export LD_LIBRARY_PATH="${ACADOS_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  export PATH="${ACADOS_DIR}/interfaces/acados_template:${PATH}"
fi

log "Acados is ready!"
