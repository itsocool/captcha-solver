#!/usr/bin/env bash
# config.sh 로 configure 한 뒤 빌드한다. build.ps1 의 리눅스 판.
#
# 사용법: ./build.sh [--config <타입>] [--clean] [--jobs N]
#
#   --config  Release(기본) / Debug / RelWithDebInfo / MinSizeRel
#   --clean   build/ 를 지우고 처음부터. config.sh 로 그대로 전달된다.
#   --jobs    동시 컴파일 잡 수 (기본: nproc)
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONFIG='Release'
CLEAN=0
JOBS="$(nproc 2>/dev/null || echo 4)"

while [ $# -gt 0 ]; do
	case "$1" in
		--config|-Config) CONFIG="${2-}"; shift 2 ;;
		--config=*)       CONFIG="${1#*=}"; shift ;;
		--clean|-Clean)   CLEAN=1; shift ;;
		--jobs|-j)        JOBS="${2-}"; shift 2 ;;
		--jobs=*)         JOBS="${1#*=}"; shift ;;
		-h|--help)        sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^#\s\?//'; exit 0 ;;
		*)                echo "알 수 없는 인자: $1" >&2; exit 2 ;;
	esac
done

# configure. cmake 를 PATH 에 얹는 것도 여기서 한다 — 별도 프로세스가 아니라 source 라서
# 이 뒤의 cmake --build 에도 그대로 유효하다(build.ps1 의 `& config.ps1` 과 같은 효과).
CONFIG_ARGS=(--config "$CONFIG")
if [ "$CLEAN" -eq 1 ]; then
	CONFIG_ARGS+=(--clean)
fi
# shellcheck source=config.sh
source "$ROOT/config.sh" "${CONFIG_ARGS[@]}"

# --config 는 단일 구성 제너레이터에서는 무시된다. Ninja Multi-Config 로 configure 한
# 경우를 위해 그대로 넘긴다.
cmake --build "$BUILD_DIR" --config "$CONFIG" -j "$JOBS"

# 단일 구성이면 build/captcha, 멀티 구성이면 build/<Config>/captcha 에 떨어진다.
EXE="$BUILD_DIR/captcha"
[ -x "$EXE" ] || EXE="$BUILD_DIR/$CONFIG/captcha"

echo ''
echo "완료 : $EXE"
