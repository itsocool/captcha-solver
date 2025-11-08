#!/usr/bin/env bash
set -euo pipefail

COMMAND=${1:-start}
PORT=5000
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
FLASK_ENV=${FLASK_ENV:-development}
DEVICE=${DEVICE:-cpu}

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
  
  # Check if we're in a production environment (Docker)
  if [ -f "/.dockerenv" ] || [ "$FLASK_ENV" = "production" ]; then
    # Use Gunicorn for production
    echo "Starting with Gunicorn (production mode)..."
    nohup "$PYTHON" -m gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 120 --access-logfile - --error-logfile - wsgi:application > "$LOG" 2>&1 &
  else
    # Use Flask dev server for development
    echo "Starting with Flask dev server..."
    nohup "$PYTHON" "$BASE_DIR/web.py" --port $PORT > "$LOG" 2>&1 &
  fi
  
  echo $! > "$PIDFILE"
  echo "Started web server with PID $(cat "$PIDFILE"), logs: $LOG"
  
  # In Docker, keep the script running to prevent container exit
  if [ -f "/.dockerenv" ] || [ "$FLASK_ENV" = "production" ]; then
    echo "Running in container mode, waiting for process to finish..."
    wait
  fi
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
