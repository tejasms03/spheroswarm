#!/usr/bin/env bash
#
# Bring the whole VLM demo up in one go: the RPC service mesh, the bench that
# drives the ball, and their React dashboard that the agent talks through.
#
#   ./demo.sh                 sim ball, everything
#   ./demo.sh --real CRXS     that Sphero over BLE
#   ./demo.sh --no-ui         skip their backend (bench + service only)
#   ./demo.sh --monitor       add a window showing what the framework sees
#   ./demo.sh --down          stop everything and free the port
#
# Each piece gets its own Terminal window rather than a background job: the
# bench is a pygame window that has to own a terminal, and when a piece dies
# the traceback needs somewhere to be read. Backgrounding all of it is how a
# session turns into "nothing happens and nobody knows why".

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="${SPHERO_FRAMEWORK:-$HOME/Downloads/Mobile-manipulation-with-VLMs-March}"
ENV_FILE="${SPHERO_ENV:-$HOME/.spheroswarm.env}"
PY=python3.13          # NOT python3: that is 3.14 here, with no cv2
PORT=5555

SOURCE=sim
ROBOT=""
WANT_UI=1
WANT_MONITOR=0

while [ $# -gt 0 ]; do
  case "$1" in
    --real)     SOURCE=0; ROBOT="${2:-}"; shift 2 ;;
    --camera)   SOURCE="${2:-0}"; shift 2 ;;
    --no-ui)    WANT_UI=0; shift ;;
    --monitor)  WANT_MONITOR=1; shift ;;
    --down)     DOWN=1; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# -- stopping ---------------------------------------------------------------

free_port() {
  # `kill $(lsof -t ...)` fails with "not enough arguments" when nothing is
  # listening, which is the normal case and not an error worth stopping for.
  local pids
  pids="$(lsof -tnP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "  freeing port $PORT (pids: $(echo "$pids" | tr '\n' ' '))"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

if [ "${DOWN:-0}" = "1" ]; then
  echo "stopping the demo"
  pkill -f "vlm.server"    2>/dev/null || true
  pkill -f "coast_test.py" 2>/dev/null || true
  pkill -f "backend/server.py" 2>/dev/null || true
  free_port
  echo "done"
  exit 0
fi

# -- checks that are cheap now and expensive at the rig ---------------------

command -v "$PY" >/dev/null || { echo "no $PY on PATH" >&2; exit 1; }

if [ ! -f "$ENV_FILE" ]; then
  echo "WARNING: no $ENV_FILE — the agent will fail with 'PORTKEY_API_KEY not set'"
else
  # Read here only to check it; each window sources it for itself so the key
  # never sits in this script's exported environment.
  grep -q PORTKEY_API_KEY "$ENV_FILE" || \
    echo "WARNING: $ENV_FILE has no PORTKEY_API_KEY"
fi

if [ "$WANT_UI" = "1" ] && [ ! -d "$FRAMEWORK" ]; then
  echo "no framework at $FRAMEWORK — set SPHERO_FRAMEWORK, or pass --no-ui" >&2
  exit 1
fi

echo "checking the framework has our files"
"$PY" -m vlm.install --check || {
  echo
  echo "  ^ run '$PY -m vlm.install' to copy them in, then start again" >&2
  exit 1
}

free_port

# -- launching --------------------------------------------------------------

win() {                                  # win <title> <cwd> <command>
  local title="$1" cwd="$2" cmd="$3"
  local script="cd $(printf '%q' "$cwd"); "
  [ -f "$ENV_FILE" ] && script+="set -a; . $(printf '%q' "$ENV_FILE"); set +a; "
  script+="echo '--- $title ---'; $cmd"
  osascript >/dev/null <<APPLESCRIPT
tell application "Terminal"
  do script "$(printf '%s' "$script" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  activate
end tell
APPLESCRIPT
  echo "  started: $title"
}

BENCH="$PY coast_test.py --source $SOURCE --rpc"
[ -n "$ROBOT" ] && BENCH="$BENCH --robot $ROBOT"

echo "bringing it up"
win "rpc service"  "$HERE" "$PY -m vlm.server"
sleep 2                                  # the bench refuses if nothing is listening
win "bench"        "$HERE" "$BENCH"

[ "$WANT_MONITOR" = "1" ] && win "monitor" "$HERE" "$PY -m vlm.monitor"
if [ "$WANT_UI" = "1" ]; then
  sleep 2
  win "dashboard"  "$FRAMEWORK" "$PY backend/server.py"
fi

cat <<EOF

  up. dashboard: http://localhost:8080   (8080, not the 8000 their docs say)

  in the bench window
    c   click the four arena corners        t   click the ball
    m   centimetre scale: drives 3s, then m again to type the tape reading
    g   go        esc   STOP        / ask the agent

  ./demo.sh --down   stops all of it
EOF
