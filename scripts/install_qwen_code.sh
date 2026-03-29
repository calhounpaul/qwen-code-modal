#!/bin/bash
set -e

# Install qwen-code CLI and configure it for the Modal coding-agent-server.
# Always disables telemetry.
#
# Usage:
#   MODAL_WORKSPACE=yourworkspace bash scripts/install_qwen_code.sh
#
# Or with a .env file in the project root containing MODAL_WORKSPACE=...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Resolve MODAL_WORKSPACE ---

# Try .env file if not already set
if [ -z "$MODAL_WORKSPACE" ] && [ -f "$PROJECT_DIR/.env" ]; then
    MODAL_WORKSPACE="$(grep -E '^MODAL_WORKSPACE=' "$PROJECT_DIR/.env" | cut -d'=' -f2- | tr -d '[:space:]')"
fi

# Interactive prompt as last resort
if [ -z "$MODAL_WORKSPACE" ]; then
    read -rp "Enter your Modal workspace name: " MODAL_WORKSPACE
fi

if [ -z "$MODAL_WORKSPACE" ]; then
    echo "Error: MODAL_WORKSPACE is required." >&2
    exit 1
fi

ENDPOINT_URL="https://${MODAL_WORKSPACE}--coding-agent-server-serve-coder.modal.run/v1"

# Get the model name from config.py
MODEL_NAME=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src/coding_agent_server')
from config import CODER_MODEL_NAME
print(CODER_MODEL_NAME)
")

# Check if multimodal model is enabled
IS_MULTIMODAL=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src/coding_agent_server')
from config import IS_MULTIMODAL_MODEL
print('true' if IS_MULTIMODAL_MODEL else 'false')
")

echo "=== Installing qwen-code CLI ==="

# Ensure npm is available (qwen-code installer requires it)
if ! command -v npm &>/dev/null; then
    echo "Error: npm is required but not installed." >&2
    echo "Install it with: sudo apt install npm" >&2
    exit 1
fi

# Install qwen-code if not present
if ! command -v qwen &>/dev/null; then
    echo "Installing qwen-code..."
    curl -fsSL https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen.sh | bash
else
    echo "qwen-code already installed: $(qwen --version 2>/dev/null || echo 'unknown version')"
fi

echo "=== Configuring qwen-code settings ==="

# Disable telemetry and configure model/endpoint settings
QWEN_SETTINGS_DIR="$HOME/.qwen"
QWEN_SETTINGS_FILE="$QWEN_SETTINGS_DIR/settings.json"

mkdir -p "$QWEN_SETTINGS_DIR"

python3 <<PYEOF
import json, os

settings_file = "$QWEN_SETTINGS_FILE"
endpoint_url = "$ENDPOINT_URL"
model_name = "$MODEL_NAME"
is_multimodal = "$IS_MULTIMODAL" == "true"

# Load existing or start fresh
if os.path.exists(settings_file):
    with open(settings_file) as f:
        d = json.load(f)
else:
    d = {}

# Always disable telemetry
d.setdefault('telemetry', {})['enabled'] = False

# Configure model
d.setdefault('model', {})['name'] = model_name

# Configure API endpoint
auth = d.setdefault('security', {}).setdefault('auth', {})
auth['baseUrl'] = endpoint_url
auth['apiKey'] = 'EMPTY'

# Configure VLM MCP server (disabled for multimodal models)
# Handle both 'mcpServers' (old) and 'mcp_servers' (new) keys
if is_multimodal:
    # Remove from both possible keys
    d.get('mcp_servers', {}).pop('vlm-analyzer', None)
    d.get('mcpServers', {}).pop('vlm-analyzer', None)
    print('Note: VLM MCP server disabled (built-in multimodal model detected)')
else:
    # Add to 'mcpServers' key (qwen-code uses this format)
    if 'mcpServers' not in d:
        d['mcpServers'] = {}
    d['mcpServers']['vlm-analyzer'] = {
        'command': 'python3',
        'args': ['$PROJECT_DIR/src/coding_agent_server/vlm_mcp_server.py'],
        'env': {
            'VLM_ENDPOINT': endpoint_url,
            'VLM_MODEL': 'Qwen/Qwen3-VL-32B-Thinking-FP8',
            'MODAL_PROXY_TOKEN_ID': os.environ.get('MODAL_PROXY_TOKEN_ID', ''),
            'MODAL_PROXY_TOKEN_SECRET': os.environ.get('MODAL_PROXY_TOKEN_SECRET', ''),
        }
    }

# Write back
with open(settings_file, 'w') as f:
    json.dump(d, f, indent=2)

print(f'Settings written to {settings_file}')
print(f'  telemetry.enabled = {d["telemetry"]["enabled"]}')
print(f'  model.name = {d["model"]["name"]}')
print(f'  security.auth.baseUrl = {d["security"]["auth"]["baseUrl"]}')
if not is_multimodal:
    print(f'  mcp_servers.vlm-analyzer = enabled')
PYEOF

# --- Write modal-env.sh for sourcing ---

MODAL_ENV_FILE="$QWEN_SETTINGS_DIR/modal-env.sh"
cat > "$MODAL_ENV_FILE" <<EOF
# Source this file to configure environment for qwen-code with Modal endpoint.
# Usage: source ~/.qwen/modal-env.sh
export OPENAI_BASE_URL="$ENDPOINT_URL"
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL="$MODEL_NAME"
EOF

echo ""
echo "=== Done ==="
echo "Workspace:  $MODAL_WORKSPACE"
echo "Model:      $MODEL_NAME"
echo "Multimodal: $IS_MULTIMODAL"
echo "Telemetry:  disabled"
echo ""
echo "Environment file written to: $MODAL_ENV_FILE"
echo ""
echo "To use:"
echo "  source ~/.qwen/modal-env.sh && qwen"
