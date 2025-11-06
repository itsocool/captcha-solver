#!/usr/bin/env bash
set -euo pipefail

COMMAND=${1:-start}
PORT=5000
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
# python 우선, 없으면 python3 사용
if command -v python >/dev/null 2>&1; then
  PYTHON=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=$(command -v python3)
else
  echo "ERROR: Neither python nor python3 found in PATH" >&2
  exit 1
fi
LOG_DIR="$BASE_DIR/logs"
LOG="$LOG_DIR/web.log"
PIDFILE="$BASE_DIR/web.pid"

mkdir -p "$LOG_DIR"

is_running() {
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if [ -z "$PID" ]; then return 1; fi
    if ps -p "$PID" > /dev/null 2>&1; then
      return 0
    else
      return 1
    fi
  fi
  return 1
}

case "$COMMAND" in
start)
  if is_running; then
    echo "Already running (PID $(cat "$PIDFILE"))"
    exit 0
  fi

  echo "Using python: $PYTHON"
  nohup "$PYTHON" "$BASE_DIR/web.py" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  echo "Started web.py with PID $(cat "$PIDFILE"), logs: $LOG"
  ;;

stop)
  if ! [ -f "$PIDFILE" ]; then
    echo "Not running (no pidfile)"
    exit 0
  fi
  PID=$(cat "$PIDFILE")
  if [ -z "$PID" ]; then
    echo "Pidfile empty, removing"; rm -f "$PIDFILE"; exit 0
  fi
  if kill "$PID" >/dev/null 2>&1; then
    echo "Sent TERM to $PID"
    rm -f "$PIDFILE"
  else
    echo "Failed to kill $PID; maybe not running; removing pidfile"
    rm -f "$PIDFILE"
  fi
  ;;

status)
  if is_running; then
    echo "Running (PID $(cat "$PIDFILE"))"
    exit 0
  else
    echo "Not running"
    exit 1
  fi
  ;;

restart)
  "$0" stop
  sleep 1
  "$0" start
  ;;

*)
  echo "Usage: $0 {start|stop|status|restart}"
  exit 2
  ;;
esac
