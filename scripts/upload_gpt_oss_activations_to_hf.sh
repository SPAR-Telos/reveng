#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_BIN="${ROOT_DIR}/.venv/bin/hf"
SOURCE_DIR="${ROOT_DIR}/outputs/activation_collection/gpt_oss_20b_boundary_v1"
REPO_ID="${1:-project-telos/gpt_oss_20b_doorkey_boundary_activations}"

# The system-level Hugging Face cache is read-only on this instance. Keep the
# token and resumable-upload metadata in a private, Git-ignored directory.
export HF_HOME="${ROOT_DIR}/.hf_home"
mkdir -p "${HF_HOME}"
chmod 700 "${HF_HOME}"

if [[ ! -x "${HF_BIN}" ]]; then
    echo "Missing Hugging Face CLI: ${HF_BIN}" >&2
    exit 1
fi
if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "Missing activation directory: ${SOURCE_DIR}" >&2
    exit 1
fi

HAS_STORED_TOKEN=false
if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    HAS_STORED_TOKEN=true
elif [[ -s "${HF_HOME}/token" || -s "${HF_HOME}/stored_tokens" ]]; then
    HAS_STORED_TOKEN=true
fi

if [[ "${HAS_STORED_TOKEN}" != true ]]; then
    echo "Enter a Hugging Face write token at the secure prompt."
    "${HF_BIN}" auth login
fi

echo "Authenticated account:"
"${HF_BIN}" auth whoami

echo "Checking permission to create or update ${REPO_ID}..."
"${HF_BIN}" repo create "${REPO_ID}" --repo-type dataset --exist-ok

echo "Uploading the unpacked directory so individual shards remain downloadable."
"${HF_BIN}" upload-large-folder \
    "${REPO_ID}" \
    "${SOURCE_DIR}" \
    --repo-type dataset \
    --num-workers 4
