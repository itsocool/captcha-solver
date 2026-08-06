#!/usr/bin/env bash
# 대법원(supreme_court) 샘플 이미지를 CLI로 모두 인식하고 정답(파일명)과 대조한다.
#
# 사용법: ./pred.sh [captcha_id]
set -uo pipefail

cd "$(dirname "$0")"

CAPTCHA_ID="${1:-supreme_court}"
SAMPLES_DIR="samples/$CAPTCHA_ID"
# 배포 번들에서는 스크립트 옆에, 개발 중에는 target/release/에 바이너리가 있다
if [ -x "./captcha-cli" ]; then
	CLI="./captcha-cli"
else
	CLI="target/release/captcha-cli"
fi

[ -x "$CLI" ] || { echo "CLI 바이너리가 없습니다: $CLI (cargo build --release)"; exit 2; }
[ -d "$SAMPLES_DIR" ] || { echo "샘플 디렉터리가 없습니다: $SAMPLES_DIR"; exit 2; }

printf '%-16s %-10s %-10s %-10s %s\n' "이미지" "정답" "예측" "신뢰도" "시간"
printf '%s\n' "--------------------------------------------------------------"

total=0
match=0

for image in "$SAMPLES_DIR"/*.png; do
	[ -e "$image" ] || continue
	name="$(basename "$image")"
	truth="${name%.png}"

	if ! output="$("$CLI" -c "$CAPTCHA_ID" -i "$image" --json 2>&1)"; then
		total=$((total + 1))
		printf '%-16s %-10s %s\n' "$name" "${truth}" "추론 실패: ${output%%$'\n'*}"
		continue
	fi
	pred="$(printf '%s' "$output" | sed -n 's/.*"prediction":"\([^"]*\)".*/\1/p')"
	conf="$(printf '%s' "$output" | sed -n 's/.*"confidence":\([0-9.eE+-]*\).*/\1/p')"
	elapsed="$(printf '%s' "$output" | sed -n 's/.*"elapsed_ms":\([0-9]*\).*/\1/p')"

	total=$((total + 1))
	if [ "$pred" = "$truth" ]; then
		match=$((match + 1))
		mark="O"
	else
		mark="X"
	fi

	printf '%-16s %-10s %-10s %-10.4f %s ms  %s\n' "$name" "$truth" "$pred" "$conf" "$elapsed" "$mark"
done

printf '%s\n' "--------------------------------------------------------------"
if [ "$total" -eq 0 ]; then
	echo "샘플 이미지가 없습니다: $SAMPLES_DIR"
	exit 2
fi
printf '%s: %d/%d 일치\n' "$CAPTCHA_ID" "$match" "$total"
[ "$match" -eq "$total" ]
