set -euo pipefail

ACADOS_DIR="${PIXI_PROJECT_ROOT}/acados"

# Setting Environment Variables
if [ -f ${ACADOS_DIR}/lib/libacados.so ]; then
  export ACADOS_SOURCE_DIR="$ACADOS_DIR"
  export ACADOS_INSTALL_DIR="$ACADOS_DIR"
  export LD_LIBRARY_PATH="${ACADOS_DIR}/lib:${LD_LIBRARY_PATH}"
  export PATH="${ACADOS_DIR}/interfaces/acados_template:${PATH}"
fi

echo "[Setup Acados] Acados is ready!"
