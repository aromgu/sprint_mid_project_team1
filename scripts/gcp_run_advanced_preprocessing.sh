#!/usr/bin/env bash
# GCP의 rhwp-python 0.8.1 FreeType ABI 충돌을 실행 프로세스에만 격리한다.

set -euo pipefail

readonly FREETYPE_LIBRARY="/lib/x86_64-linux-gnu/libfreetype.so.6"

if [[ -n "${LD_PRELOAD:-}" ]]; then
  echo "오류: 예상하지 못한 LD_PRELOAD가 이미 설정돼 있습니다." >&2
  echo "전역 설정을 제거한 새 셸에서 다시 실행하세요." >&2
  exit 2
fi

if [[ ! -r "${FREETYPE_LIBRARY}" ]]; then
  echo "오류: GCP 시스템 FreeType을 읽을 수 없습니다: ${FREETYPE_LIBRARY}" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "오류: uv를 찾을 수 없습니다. source \$HOME/.local/bin/env를 실행하세요." >&2
  exit 2
fi

# uv 자체에는 preload하지 않고 실제 Python 자식 프로세스에만 적용한다.
uv run env \
  LD_PRELOAD="${FREETYPE_LIBRARY}" \
  python -c 'import rhwp; print("rhwp preflight: ready")'

exec uv run env \
  LD_PRELOAD="${FREETYPE_LIBRARY}" \
  python -m scripts.run_advanced_preprocessing "$@"
