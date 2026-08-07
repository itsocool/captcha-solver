#!/usr/bin/env bash
# CMake configure. cmake 를 찾아 PATH 에 얹고 build/ 를 만든다.
# config.ps1 의 리눅스 판. build.sh 가 이 스크립트를 source 한다.
#
# 사용법: ./config.sh [--config <타입>] [--clean]
#
#   --config  Release(기본) / Debug / RelWithDebInfo / MinSizeRel
#   --clean   build/ 를 지우고 처음부터 configure. 제너레이터를 바꿨을 때 필요하다.
#
# Windows 판과 다른 점:
#   - vswhere 대신 흔한 설치 경로를 훑는다(전역 PATH 는 건드리지 않고 이 셸에만 얹는다).
#   - 리눅스 기본 제너레이터(Unix Makefiles / Ninja)는 단일 구성이라 빌드 타입을
#     configure 시점에 정해야 한다. 그래서 config.ps1 에 없는 --config 를 받는다.
#   - -A x64 는 Visual Studio 제너레이터 전용이라 주지 않는다.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"

CONFIG='Release'
CLEAN=0

while [ $# -gt 0 ]; do
	case "$1" in
		--config|-Config) CONFIG="${2-}"; shift 2 ;;
		--config=*)       CONFIG="${1#*=}"; shift ;;
		--clean|-Clean)   CLEAN=1; shift ;;
		-h|--help)        sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^#\s\?//'; exit 0 ;;
		*)                echo "알 수 없는 인자: $1" >&2; exit 2 ;;
	esac
done

case "$CONFIG" in
	Release|Debug|RelWithDebInfo|MinSizeRel) ;;
	*) echo "--config 는 Release / Debug / RelWithDebInfo / MinSizeRel 중 하나여야 합니다: '$CONFIG'" >&2; exit 2 ;;
esac

# cmake 를 PATH 에 얹는다. PATH 는 이미 export 되어 있으므로 자식 프로세스에도 전달된다.
add_cmake_to_path() {
	command -v cmake >/dev/null 2>&1 && return 0

	local dir
	for dir in /usr/local/bin /opt/cmake/bin /snap/bin "$HOME/.local/bin" /opt/cmake-*/bin; do
		if [ -x "$dir/cmake" ]; then
			PATH="$dir:$PATH"
			return 0
		fi
	done

	cat >&2 <<'EOF'
cmake 를 찾을 수 없습니다. 배포판에 맞게 설치하세요:
  Debian/Ubuntu : sudo apt install cmake build-essential
  Fedora/RHEL   : sudo dnf install cmake gcc-c++ make
  Arch          : sudo pacman -S cmake base-devel
  배포판 cmake 가 3.24 미만이면 https://cmake.org/download 의 tarball 을 /opt 에 풀어도 된다.
EOF
	exit 1
}

add_cmake_to_path
echo "cmake : $(command -v cmake) ($(cmake --version | head -n1 | awk '{print $3}'))"

if [ "$CLEAN" -eq 1 ] && [ -d "$BUILD_DIR" ]; then
	echo "clean : $BUILD_DIR 삭제"
	rm -rf -- "$BUILD_DIR"
fi

# 제너레이터는 지정하지 않는다 — CMake 가 알아서 Ninja/Make 를 고르므로 환경이 바뀌어도
# 이 스크립트를 고칠 필요가 없다.
status=0
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG" || status=$?
if [ "$status" -ne 0 ]; then
	echo ''
	echo 'configure 실패. 자주 있는 원인:'
	echo '  - C++ 컴파일러 없음      : sudo apt install build-essential (GCC 9+ / Clang 9+ 필요)'
	echo "  - 제너레이터/빌드 타입 변경 : ./config.sh --clean 으로 build/ 를 지우고 재실행"
	echo '  - 최초 configure 는 ONNX Runtime(~120MB)을 받으므로 네트워크가 필요합니다'
	echo "cmake configure 실패 (exit $status)" >&2
	exit "$status"
fi
