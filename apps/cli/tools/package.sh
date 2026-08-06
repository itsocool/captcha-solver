#!/usr/bin/env bash
# 배포용 tar.gz 생성: 실행 파일 + 모델 + 샘플 + 문서
#
# 사용법: tools/package.sh
# 산출물: dist/captcha-cli-<version>-linux-<arch>.tar.gz
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -1)"
ARCH="$(uname -m)"
NAME="captcha-cli-${VERSION}-linux-${ARCH}"
STAGE="dist/${NAME}"

[ -x target/release/captcha-cli ] || { echo "먼저 cargo build --release 를 실행하세요"; exit 2; }

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp target/release/captcha-cli "$STAGE/"
cp README.md pred.sh pred.cmd pred.ps1 "$STAGE/"
cp -r models "$STAGE/"
cp -r samples "$STAGE/"

tar -czf "dist/${NAME}.tar.gz" -C dist "$NAME"
rm -rf "$STAGE"

echo "dist/${NAME}.tar.gz  ($(du -h "dist/${NAME}.tar.gz" | cut -f1))"
