#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- Bootstrap venv + modal if needed ---

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet "modal>=0.73" aiohttp
    echo "Venv ready."
fi

# --- Ensure modal token exists ---

if ! "$VENV_DIR/bin/modal" token list &>/dev/null; then
    echo "No Modal token found. Launching authentication..."
    "$VENV_DIR/bin/modal" token new
fi

# --- Load .env if present ---

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$SCRIPT_DIR/.env"
    set +a
fi

# --- Resolve MODAL_WORKSPACE ---

if [ -z "$MODAL_WORKSPACE" ]; then
    read -rp "Enter your Modal workspace name: " MODAL_WORKSPACE
fi

if [ -z "$MODAL_WORKSPACE" ]; then
    echo "Error: MODAL_WORKSPACE is required." >&2
    echo "Set it in .env or pass as: MODAL_WORKSPACE=yourworkspace ./run.sh [command]" >&2
    exit 1
fi

# --- Env vars for qwen-code / OpenAI-compatible clients ---

export OPENAI_BASE_URL="https://${MODAL_WORKSPACE}--coding-agent-server-serve-coder.modal.run/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_MODEL="unsloth/Qwen3-Coder-Next-FP8-Dynamic"

export VLM_BASE_URL="https://${MODAL_WORKSPACE}--coding-agent-server-serve-vlm.modal.run/v1"
export VLM_MODEL="Qwen/Qwen3-VL-32B-Thinking-FP8"

# --- Dispatch command ---

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
    deploy)
        echo "Deploying coding-agent-server..."
        "$VENV_DIR/bin/modal" deploy "$SCRIPT_DIR/src/coding_agent_server/deploy.py" "$@"
        ;;
    smoke)
        echo "Running smoke test (modal run)..."
        "$VENV_DIR/bin/modal" run "$SCRIPT_DIR/src/coding_agent_server/deploy.py" "$@"
        ;;
    logs)
        "$VENV_DIR/bin/modal" app logs coding-agent-server "$@"
        ;;
    install)
        # Ensure npm is available (qwen-code installer requires it)
        if ! command -v npm &>/dev/null; then
            echo "Error: npm is required but not installed." >&2
            echo "Install it with: sudo apt install npm" >&2
            exit 1
        fi

        # Install qwen-code if needed
        if ! command -v qwen &>/dev/null; then
            echo "Installing qwen-code..."
            curl -fsSL https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen.sh | bash
        else
            echo "qwen-code already installed: $(qwen --version 2>/dev/null || echo 'unknown')"
        fi

        # Write qodal wrapper to ~/.local/bin
        mkdir -p "$HOME/.local/bin"
        cat > "$HOME/.local/bin/qodal" <<EOF
#!/bin/bash
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_MODEL="$OPENAI_MODEL"
exec qwen "\$@"
EOF
        chmod +x "$HOME/.local/bin/qodal"
        echo ""
        echo "Installed: ~/.local/bin/qodal"
        echo "  Endpoint: $OPENAI_BASE_URL"
        echo "  Model:    $OPENAI_MODEL"

        # Install MCP server dependencies into project venv
        echo ""
        echo "Installing VLM MCP server dependencies..."
        "$VENV_DIR/bin/pip" install --quiet "mcp[cli]" httpx

        # Register vlm-analyzer MCP server in qwen-code settings
        echo "Registering vlm-analyzer MCP server..."
        QWEN_SETTINGS_FILE="$HOME/.qwen/settings.json"
        mkdir -p "$HOME/.qwen"

        MCP_SERVER_SCRIPT="$SCRIPT_DIR/src/coding_agent_server/vlm_mcp_server.py"
        VENV_PYTHON="$VENV_DIR/bin/python"

        python3 -c "
import json, os

settings_file = '$QWEN_SETTINGS_FILE'
vlm_endpoint = '$VLM_BASE_URL'
vlm_model = '$VLM_MODEL'
mcp_script = '$MCP_SERVER_SCRIPT'
venv_python = '$VENV_PYTHON'

# Load existing or start fresh
if os.path.exists(settings_file):
    with open(settings_file) as f:
        d = json.load(f)
else:
    d = {}

# Add/update mcpServers entry (merge, don't clobber)
mcp_servers = d.setdefault('mcpServers', {})
mcp_servers['vlm-analyzer'] = {
    'command': venv_python,
    'args': [mcp_script],
    'env': {
        'VLM_ENDPOINT': vlm_endpoint,
        'VLM_MODEL': vlm_model,
    },
}

with open(settings_file, 'w') as f:
    json.dump(d, f, indent=2)

print(f'Registered vlm-analyzer MCP server in {settings_file}')
"

        echo ""
        echo "Run 'qodal' from any directory to use qwen-code with the Modal endpoint."
        echo "VLM MCP server (vlm-analyzer) is registered and will be available in qwen-code."
        ;;
    qwen)
        echo "Endpoint: $OPENAI_BASE_URL"
        echo "Model:    $OPENAI_MODEL"
        exec qwen "$@"
        ;;
    test)
        ENDPOINT_URL="https://${MODAL_WORKSPACE}--coding-agent-server-serve-coder.modal.run" \
        VLM_ENDPOINT_URL="https://${MODAL_WORKSPACE}--coding-agent-server-serve-vlm.modal.run" \
            "$VENV_DIR/bin/python" -m pytest "$SCRIPT_DIR/tests/test_health.py" -v "$@"
        ;;
    env)
        echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
        echo "OPENAI_API_KEY=$OPENAI_API_KEY"
        echo "OPENAI_MODEL=$OPENAI_MODEL"
        echo "VLM_BASE_URL=$VLM_BASE_URL"
        echo "VLM_MODEL=$VLM_MODEL"
        echo "MODAL_WORKSPACE=$MODAL_WORKSPACE"
        ;;
    shell)
        echo "Activating venv with Modal endpoint env vars..."
        echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
        exec "$VENV_DIR/bin/python" "$@"
        ;;
    help|*)
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  install  Install qwen-code + 'qodal' wrapper + VLM MCP server"
        echo "  deploy   Deploy the server to Modal (coder + VLM)"
        echo "  smoke    Run smoke test via modal run"
        echo "  logs     Tail Modal app logs"
        echo "  qwen     Launch qwen-code CLI with endpoint configured"
        echo "  test     Run pytest health checks against live endpoints"
        echo "  env      Print configured env vars (coder + VLM)"
        echo "  shell    Start Python with venv + env vars"
        echo ""
        echo "Config: set MODAL_WORKSPACE in .env or environment"
        ;;
esac
