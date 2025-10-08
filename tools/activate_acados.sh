set -euo pipefail

ACADOS_DIR="${PIXI_PROJECT_ROOT}/acados"

# Install Acados Python interface
if [ -d "${ACADOS_DIR}/interfaces/acados_template" ] && ! pip show acados-template >/dev/null 2>&1; then
  echo "[Setup Acados] Installing acados Python interface..."
  pip install -e ${ACADOS_DIR}/interfaces/acados_template
fi

# Setting Environment Variables
if [ -f ${ACADOS_DIR}/lib/libacados.so ]; then
  echo "[Setup Acados] Exporting acados environment variables..."
  export ACADOS_SOURCE_DIR="$ACADOS_DIR"
  export ACADOS_INSTALL_DIR="$ACADOS_DIR"
  export LD_LIBRARY_PATH="${ACADOS_DIR}/lib:${LD_LIBRARY_PATH}"
  export PATH="${ACADOS_DIR}/interfaces/acados_template:${PATH}"
fi

echo "[Setup Acados] Acados is ready!"
