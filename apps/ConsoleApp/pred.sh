#!/usr/bin/env bash
# 샘플 이미지를 captcha 로 인식하고 정답(파일명)과 대조한다.
# pred.ps1 의 리눅스 판. 기본은 captcha_data 에서 100장을 무작위로 뽑는다.
#
# 사용법: ./pred.sh [captcha_id] [--count N] [--seed N] [--config <타입>]
#
#   captcha_id  기본 supreme_court
#   --count     장수 (기본 100). 전체보다 크면 전부 본다.
#   --seed      난수 시드. 같은 표본을 재현할 때.
#   --config    Release(기본) / Debug / RelWithDebInfo / MinSizeRel
set -uo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

CAPTCHA_ID='supreme_court'
COUNT=100
SEED=''
CONFIG='Release'

while [ $# -gt 0 ]; do
	case "$1" in
		--count|-Count)   COUNT="${2-}"; shift 2 ;;
		--count=*)        COUNT="${1#*=}"; shift ;;
		--seed|-Seed)     SEED="${2-}"; shift 2 ;;
		--seed=*)         SEED="${1#*=}"; shift ;;
		--config|-Config) CONFIG="${2-}"; shift 2 ;;
		--config=*)       CONFIG="${1#*=}"; shift ;;
		-h|--help)        sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^#\s\?//'; exit 0 ;;
		-*)               echo "알 수 없는 인자: $1" >&2; exit 2 ;;
		*)                CAPTCHA_ID="$1"; shift ;;
	esac
done

# 배포 번들에서는 스크립트 옆에, 개발 중에는 build/ (멀티 구성이면 build/<Config>/) 에 있다
EXE='./captcha'
[ -x "$EXE" ] || EXE='./build/captcha'
[ -x "$EXE" ] || EXE="./build/$CONFIG/captcha"
if [ ! -x "$EXE" ]; then
	echo "실행 파일이 없습니다: ./build/captcha (./build.sh --config $CONFIG)"
	exit 2
fi

# 이미지 출처: captcha_data 의 pred 세트를 우선 쓰고, 없으면 cli 샘플 10장으로 폴백한다.
# rev 는 메타데이터에 적힌 것을 쓰고, 메타가 없으면 가장 큰 번호를 고른다.
resolve_image_dir() {
	local meta="../cli/models/$CAPTCHA_ID.meta.json" rev='' base newest dir

	if [ -f "$meta" ]; then
		# jq 없이 "rev": 0 또는 "rev": "0" 을 뽑는다.
		rev="$(sed -n 's/.*"rev"[[:space:]]*:[[:space:]]*"\?\([0-9][0-9]*\).*/\1/p' "$meta" | head -n1)"
	fi

	base="../../captcha_data/$CAPTCHA_ID"
	if [ -d "$base" ]; then
		if [ -n "$rev" ] && [ -d "$base/$rev/images/pred" ]; then
			printf '%s\n' "$base/$rev/images/pred"
			return 0
		fi
		newest="$(ls -1 "$base" 2>/dev/null | grep -Ex '[0-9]+' | sort -n | tail -n1)"
		if [ -n "$newest" ] && [ -d "$base/$newest/images/pred" ]; then
			printf '%s\n' "$base/$newest/images/pred"
			return 0
		fi
	fi

	for dir in "samples/$CAPTCHA_ID" "../cli/samples/$CAPTCHA_ID"; do
		if [ -d "$dir" ]; then
			printf '%s\n' "$dir"
			return 0
		fi
	done
	return 1
}

IMAGE_DIR="$(resolve_image_dir)" || {
	echo "이미지 디렉터리를 찾을 수 없습니다: captcha_data/$CAPTCHA_ID 또는 ../cli/samples/$CAPTCHA_ID"
	exit 2
}

mapfile -t ALL < <(find "$IMAGE_DIR" -maxdepth 1 -type f -name '*.png' | sort)
if [ "${#ALL[@]}" -eq 0 ]; then
	echo "이미지가 없습니다: $IMAGE_DIR"
	exit 2
fi

# 시드를 주지 않으면 매번 다른 표본. rand() 값은 정수로 찍어 로케일(소수점 , 아님) 영향을 없앤다.
[ -n "$SEED" ] || SEED="$RANDOM$$"
mapfile -t IMAGES < <(
	printf '%s\n' "${ALL[@]}" |
		awk -v seed="$SEED" 'BEGIN { srand(seed) } { printf "%d\t%s\n", int(rand() * 2147483647), $0 }' |
		sort -k1,1n | cut -f2- | head -n "$COUNT"
)

printf '출처 : %s  (%d장 중 %d장 무작위)\n' "$(readlink -f "$IMAGE_DIR")" "${#ALL[@]}" "${#IMAGES[@]}"
echo ''

# captcha 는 --json 이 없어 예측 문자열만 내보낸다. 신뢰도 열이 없는 대신 시간은 프로세스
# 전체 실행 시간이다(ONNX 세션 초기화 포함이라 Rust CLI 의 내부 측정값보다 크게 나온다).
ROW_FORMAT='%-16s %-10s %-10s %-8s %s'

# 헤더는 printf 패딩을 쓰지 않는다. 한글은 화면에서 2칸을 차지하는데 %-16s 같은 패딩은
# 바이트 수로 세기 때문에 데이터 행과 어긋난다. 데이터 행의 열 위치(17/28/39/48)에
# 맞춰 공백을 직접 넣었다.
echo '이미지           정답       예측       시간     결과'
echo '------------------------------------------------------'

# date +%s%3N 는 쓰지 않는다 — uutils coreutils(우분투 26.04 기본)의 date 는 %3N 의 폭
# 지정자를 무시하고 나노초 9자리를 그대로 찍는다. %s%N 으로 받아 직접 ms 로 나눈다.
now_ns() { date +%s%N; }

total=0
match=0
start_total="$(now_ns)"

for image in "${IMAGES[@]}"; do
	name="$(basename -- "$image")"
	truth="${name%.png}"
	total=$((total + 1))

	start="$(now_ns)"
	out="$("$EXE" "-c=$CAPTCHA_ID" "-i=$image" 2>&1)"
	rc=$?
	elapsed="$(( ($(now_ns) - start) / 1000000 )) ms"

	if [ "$rc" -ne 0 ]; then
		printf "$ROW_FORMAT  %s\n" "$name" "$truth" '-' "$elapsed" '❌' "${out%%$'\n'*}"
		continue
	fi

	if [ "$out" = "$truth" ]; then
		match=$((match + 1))
		mark='✅'
	else
		mark='❌'
	fi
	printf "$ROW_FORMAT\n" "$name" "$truth" "$out" "$elapsed" "$mark"
done

total_ms=$(( ($(now_ns) - start_total) / 1000000 ))
echo '------------------------------------------------------'
# 헤더와 같은 이유로 라벨 뒤 공백을 직접 맞췄다(값은 모두 12번째 칸에서 시작).
printf '캡차        %s\n' "$CAPTCHA_ID"
printf '일치        %d/%d\n' "$match" "$total"
awk -v m="$match" -v t="$total" 'BEGIN { printf "일치율      %.1f%%\n", m * 100 / t }'
awk -v ms="$total_ms" -v t="$total" \
	'BEGIN { printf "총 소요     %.1f초  (평균 %d ms)\n", ms / 1000, ms / t }'

[ "$match" -eq "$total" ]
