#!/usr/bin/env bash
#
# captchaSolver Spring Boot 개발서버 관리
#
#   ./devserver.sh start|stop|restart|status
#
# application.yml 의 모델·DB 경로가 상대경로(../../apps/cli/models 등)라서 이 스크립트는
# 항상 자기 위치(apps/springBoot)로 이동한 뒤 maven 을 띄운다.
#
# 환경변수
#   PORT           서버 포트 (기본 5000)
#   START_TIMEOUT  기동 대기 한계, 초 (기본 120)
#   STOP_TIMEOUT   graceful 종료 대기 한계, 초 (기본 20)
#   MVN            maven 실행 파일 (기본 mvn)

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$APP_DIR/.dev"
PID_FILE="$STATE_DIR/server.pid"
LOG_FILE="$STATE_DIR/server.log"

PORT="${PORT:-5000}"
START_TIMEOUT="${START_TIMEOUT:-120}"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"
MVN="${MVN:-mvn}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 상태 조회

# 저장된 프로세스 그룹 ID. 없으면 빈 문자열.
read_pgid() {
    [[ -f $PID_FILE ]] || return 0
    local pgid
    pgid="$(tr -dc '0-9' < "$PID_FILE")"
    [[ -n $pgid && $pgid -gt 1 ]] || return 0
    printf '%s' "$pgid"
}

# 저장된 프로세스 그룹에 살아있는 멤버가 있는가.
group_alive() {
    local pgid="$1"
    [[ -n $pgid ]] || return 1
    kill -0 -- "-${pgid}" 2>/dev/null
}

# PORT 를 LISTEN 중인 프로세스 PID 들. 없으면 아무것도 출력하지 않는다.
# 매치가 없을 때 grep 이 1 을 반환하는데, pipefail + set -e 조합에서 호출부의
# 변수 대입이 통째로 실패하므로 성공으로 흡수한다.
port_listener_pids() {
    ss -ltnpH "sport = :${PORT}" 2>/dev/null \
        | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
}

port_in_use() {
    [[ -n "$(port_listener_pids)" ]]
}

# 위 PID 들을 메시지·kill 에 바로 쓸 수 있게 공백 구분 한 줄로.
port_listener_list() {
    local pids
    pids="$(port_listener_pids | tr '\n' ' ')"
    printf '%s' "${pids% }"
}

health_body() {
    curl -fsS -m 3 "$HEALTH_URL" 2>/dev/null
}

health_ok() {
    health_body >/dev/null
}

# 죽은 프로세스가 남긴 PID 파일 정리.
clear_stale_pid() {
    local pgid
    pgid="$(read_pgid)"
    if [[ -n $pgid ]] && ! group_alive "$pgid"; then
        rm -f "$PID_FILE"
    fi
}

tail_log() {
    if [[ -s $LOG_FILE ]]; then
        warn "--- ${LOG_FILE} (마지막 ${1:-30}줄) ---"
        tail -n "${1:-30}" "$LOG_FILE" >&2
        warn "--- 로그 끝 ---"
    fi
}

# ---------------------------------------------------------------- start

cmd_start() {
    mkdir -p "$STATE_DIR"
    clear_stale_pid

    local pgid
    pgid="$(read_pgid)"
    if group_alive "$pgid"; then
        log "이미 실행 중입니다 (프로세스 그룹 ${pgid}, 포트 ${PORT})."
        return 0
    fi

    command -v "$MVN" >/dev/null 2>&1 \
        || die "'${MVN}' 을 찾을 수 없습니다. maven 을 설치하거나 MVN 환경변수로 경로를 지정하세요."

    if port_in_use; then
        die "포트 ${PORT} 를 이미 다른 프로세스가 쓰고 있습니다 (PID: $(port_listener_list)). 이 스크립트가 띄운 서버가 아니므로 직접 정리해 주세요."
    fi

    log "기동합니다 (포트 ${PORT}) ..."
    : > "$LOG_FILE"

    # setsid 로 새 프로세스 그룹에 띄운다. spring-boot:run 은 애플리케이션을 자식 JVM 으로
    # fork 하므로, 그룹 단위로 시그널을 보내야 자식까지 함께 정리된다.
    cd "$APP_DIR"
    setsid "$MVN" spring-boot:run \
        "-Dspring-boot.run.arguments=--server.port=${PORT}" \
        >> "$LOG_FILE" 2>&1 &
    local leader=$!

    # 실제 PGID 를 커널에서 읽어온다 (setsid 가 fork 했을 가능성까지 흡수).
    pgid="$(ps -o pgid= -p "$leader" 2>/dev/null | tr -dc '0-9' || true)"
    [[ -n $pgid ]] || pgid="$leader"
    printf '%s\n' "$pgid" > "$PID_FILE"

    # "떴다" 의 근거는 로그 문자열이 아니라 /health 실제 응답이다.
    local waited=0
    while (( waited < START_TIMEOUT )); do
        if health_ok; then
            log "기동 완료 — http://localhost:${PORT} (프로세스 그룹 ${pgid}, ${waited}초)"
            log "로그: ${LOG_FILE}"
            return 0
        fi
        if ! group_alive "$pgid"; then
            rm -f "$PID_FILE"
            tail_log 30
            die "기동 중 프로세스가 종료됐습니다."
        fi
        sleep 1
        waited=$(( waited + 1 ))
    done

    tail_log 30
    warn "프로세스는 살아있지만 ${START_TIMEOUT}초 안에 ${HEALTH_URL} 이 응답하지 않았습니다."
    die "기동 확인 실패. 정리하려면 '$0 stop' 을 실행하세요."
}

# ---------------------------------------------------------------- stop

# 프로세스 그룹에 시그널을 보내고 사라질 때까지 기다린다.
signal_group_and_wait() {
    local pgid="$1" sig="$2" timeout="$3" waited=0

    kill "-${sig}" -- "-${pgid}" 2>/dev/null || true
    while (( waited < timeout )); do
        group_alive "$pgid" || return 0
        sleep 1
        waited=$(( waited + 1 ))
    done
    group_alive "$pgid" && return 1 || return 0
}

cmd_stop() {
    local pgid stopped_something=0
    pgid="$(read_pgid)"

    if group_alive "$pgid"; then
        stopped_something=1
        log "종료합니다 (프로세스 그룹 ${pgid}) ..."
        if ! signal_group_and_wait "$pgid" TERM "$STOP_TIMEOUT"; then
            warn "${STOP_TIMEOUT}초 안에 안 끝나서 강제 종료합니다."
            signal_group_and_wait "$pgid" KILL 10 \
                || warn "프로세스 그룹 ${pgid} 가 여전히 남아있습니다."
        fi
    fi
    rm -f "$PID_FILE"

    # 안전망 — PGID 기록이 어긋났거나 이전 실행이 남긴 프로세스가 포트를 물고 있을 수 있다.
    if port_in_use; then
        local pids
        pids="$(port_listener_list)"
        warn "포트 ${PORT} 가 아직 잡혀 있습니다 (PID: ${pids}). 잔여 프로세스를 정리합니다."
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
        local waited=0
        while (( waited < 10 )) && port_in_use; do sleep 1; waited=$(( waited + 1 )); done
        if port_in_use; then
            pids="$(port_listener_list)"
            # shellcheck disable=SC2086
            kill -KILL $pids 2>/dev/null || true
            sleep 1
        fi
        port_in_use && die "포트 ${PORT} 를 비우지 못했습니다 (PID: $(port_listener_list))."
        stopped_something=1
    fi

    if (( stopped_something )); then
        log "종료 완료."
    else
        log "실행 중인 서버가 없습니다."
    fi
}

# ---------------------------------------------------------------- restart / status

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_status() {
    local pgid listeners body
    pgid="$(read_pgid)"
    listeners="$(port_listener_list)"

    if group_alive "$pgid"; then
        log "프로세스  실행 중 (프로세스 그룹 ${pgid})"
    elif [[ -n $pgid ]]; then
        log "프로세스  없음 (PID 파일에 ${pgid} 가 남아있으나 죽은 상태)"
    else
        log "프로세스  없음"
    fi

    if [[ -n $listeners ]]; then
        log "포트      ${PORT} LISTEN (PID: ${listeners})"
    else
        log "포트      ${PORT} 사용 안 함"
    fi

    if body="$(health_body)"; then
        log "health    정상 — ${body}"
    else
        log "health    응답 없음 (${HEALTH_URL})"
    fi

    log "로그      ${LOG_FILE}"

    group_alive "$pgid" && health_ok
}

# ---------------------------------------------------------------- 진입점

usage() {
    cat <<EOF
사용법: $(basename "$0") {start|stop|restart|status}

  start     개발서버를 백그라운드로 띄우고 /health 응답까지 기다린다
  stop      프로세스 그룹 전체를 종료하고 포트를 비운다
  restart   stop 후 start
  status    프로세스 / 포트 / health 를 각각 확인한다

환경변수: PORT(기본 ${PORT}) START_TIMEOUT(${START_TIMEOUT}) STOP_TIMEOUT(${STOP_TIMEOUT}) MVN(${MVN})
EOF
}

main() {
    case "${1-}" in
        start)   cmd_start ;;
        stop)    cmd_stop ;;
        restart) cmd_restart ;;
        status)  cmd_status ;;
        -h|--help|help) usage ;;
        "")      usage; exit 2 ;;
        *)       warn "알 수 없는 명령: $1"; usage; exit 2 ;;
    esac
}

main "$@"
