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
MODEL_NAME="unsloth/Qwen3-Coder-Next-FP8-Dynamic"

echo "=== Installing qwen-code CLI ==="

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

python3 -c "
import json, os

settings_file = '$QWEN_SETTINGS_FILE'
endpoint_url = '$ENDPOINT_URL'
model_name = '$MODEL_NAME'

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

# Write back
with open(settings_file, 'w') as f:
    json.dump(d, f, indent=2)

print(f'Settings written to {settings_file}')
print(f'  telemetry.enabled = {d[\"telemetry\"][\"enabled\"]}')
print(f'  model.name = {d[\"model\"][\"name\"]}')
print(f'  security.auth.baseUrl = {d[\"security\"][\"auth\"][\"baseUrl\"]}')
"

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
echo "Endpoint:   $ENDPOINT_URL"
echo "Telemetry:  disabled"
echo ""
echo "Environment file written to: $MODAL_ENV_FILE"
echo ""
echo "To use:"
echo "  source ~/.qwen/modal-env.sh && qwen"
