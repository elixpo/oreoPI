#!/bin/sh
set -eu

oreo-daemon &
daemon_pid=$!

stop_daemon() {
    elixpo daemon stop >/dev/null 2>&1 \
        || kill -TERM "$daemon_pid" 2>/dev/null \
        || true
}

trap stop_daemon TERM INT

status=0
while kill -0 "$daemon_pid" 2>/dev/null; do
    if wait "$daemon_pid"; then
        status=0
    else
        status=$?
    fi
done

exit "$status"
