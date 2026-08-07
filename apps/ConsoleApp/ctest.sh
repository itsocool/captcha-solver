#!/usr/bin/env bash
# build.sh 로 빌드한 뒤 테스트를 돌린다. ctest.ps1 의 리눅스 판.
#
#   decode  - CTC prefix beam search 단위 테스트
#   samples - apps/cli/samples/<id>/ 를 인식해 파일명(정답)과 대조
#
# 사용법: ./ctest.sh [--config <타입>] [--clean] [--filter <정규식>]
#
#   --config  Release(기본) / Debug / RelWithDebInfo / MinSizeRel
#   --clean   build/ 를 지우고 처음부터.
#   --filter  테스트 이름 정규식 (ctest -R). 예: --filter decode
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONFIG='Release'
CLEAN=0
FILTER=''

while [ $# -gt 0 ]; do
	case "$1" in
		--config|-Config) CONFIG="${2-}"; shift 2 ;;
		--config=*)       CONFIG="${1#*=}"; shift ;;
		--clean|-Clean)   CLEAN=1; shift ;;
		--filter|-Filter|-R) FILTER="${2-}"; shift 2 ;;
		--filter=*)       FILTER="${1#*=}"; shift ;;
		-h|--help)        sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^#\s\?//'; exit 0 ;;
		*)                echo "알 수 없는 인자: $1" >&2; exit 2 ;;
	esac
done

BUILD_ARGS=(--config "$CONFIG")
if [ "$CLEAN" -eq 1 ]; then
	BUILD_ARGS+=(--clean)
fi
# source 라서 config.sh 가 PATH 에 얹은 cmake/ctest 가 아래에서도 유효하다.
# shellcheck source=build.sh
source "$ROOT/build.sh" "${BUILD_ARGS[@]}"

# -C 는 단일 구성 제너레이터에서는 무시되지만, 멀티 구성으로 configure 한 경우를 위해 남긴다.
CTEST_ARGS=(--test-dir "$BUILD_DIR" -C "$CONFIG" --output-on-failure)
if [ -n "$FILTER" ]; then
	CTEST_ARGS+=(-R "$FILTER")
fi

echo ''
status=0
ctest "${CTEST_ARGS[@]}" || status=$?
if [ "$status" -ne 0 ]; then
	echo "테스트 실패 (exit $status)" >&2
	exit "$status"
fi
