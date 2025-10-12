#!/bin/bash
set -euo pipefail

# Constants
readonly DEFAULT_NAMESPACE="tb2_system"

usage() {
    cat << EOF
Usage: $(basename "$0") [namespace]

Arguments:
    namespace  Turtlebot namespace to stop (default: ${DEFAULT_NAMESPACE})
EOF
}

# Get namespace from argument or use default
namespace="${1:-${DEFAULT_NAMESPACE}}"

# If inside tmux session, get the current session name
current_session=""
if [[ -n "${TMUX:-}" ]]; then
    current_session=$(tmux display-message -p '#S')
fi

# Function to gracefully stop a tmux session
stop_session() {
    local session="$1"
    
    # Check if session exists
    if ! tmux has-session -t "$session" 2>/dev/null; then
        echo "No active session found for namespace: $session"
        return 0
    fi

    echo "Stopping session: $session"
    
    # Send Ctrl+C to all windows in the session
    while read -r window_index; do
        tmux send-keys -t "${session}:${window_index}" C-c
        sleep 0.1
    done < <(tmux list-windows -t "$session" -F "#{window_index}")

    # Kill the session if it's not the current one
    if [[ "$session" != "$current_session" ]]; then
        tmux kill-session -t "$session" 2>/dev/null || true
    fi
}

# Stop the specified session
stop_session "$namespace"

# If we're in a tmux session and it's the one we just stopped, kill it
if [[ -n "$current_session" && "$current_session" == "$namespace" ]]; then
    sleep 0.5  # Give processes time to clean up
    tmux kill-session -t "$current_session" 2>/dev/null || true
fi