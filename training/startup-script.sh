#!/bin/bash
# Spot VM起動時に実行されるスクリプト。
# clone -> 依存インストール -> 学習 -> GCSへ結果アップロード -> 自己削除、まで一気通貫で行う。
# 進行状況は `gcloud compute instances get-serial-port-output` で確認できる。
set -uo pipefail

METADATA_URL="http://metadata.google.internal/computeMetadata/v1"
INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" "$METADATA_URL/instance/name")
ZONE=$(curl -s -H "Metadata-Flavor: Google" "$METADATA_URL/instance/zone" | cut -d/ -f4)
REPO_URL=$(curl -s -H "Metadata-Flavor: Google" "$METADATA_URL/instance/attributes/REPO_URL")
GCS_BUCKET=$(curl -s -H "Metadata-Flavor: Google" "$METADATA_URL/instance/attributes/GCS_BUCKET")
TRAIN_CMD=$(curl -s -H "Metadata-Flavor: Google" "$METADATA_URL/instance/attributes/TRAIN_CMD")

self_delete() {
  echo "=== self-delete: ${INSTANCE_NAME} (zone=${ZONE}) ==="
  gcloud compute instances delete "${INSTANCE_NAME}" --zone="${ZONE}" --quiet
}
# 正常終了・異常終了に関わらずVMを残さない（コスト安全策）
trap self_delete EXIT

echo "=== cloning ${REPO_URL} ==="
apt-get update -y
apt-get install -y python3-pip python3-venv git
git clone --depth 1 "${REPO_URL}" /app
cd /app

echo "=== installing dependencies ==="
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
# requirements.txtのtorch>=2.4は無指定だとCUDA同梱版（数GB）を引っ張り、
# ディスク容量を圧迫するため、CPU専用版を先にインストールしておく
# （要求バージョンを満たすのでrequirements.txt側では再インストールされない）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
df -h /

echo "=== downloading existing champion state from ${GCS_BUCKET} ==="
# 新しいVMはgit cloneしただけなのでrunsが空（runs/はgitignore対象）。
# 既存のchampionと比較するため、GCS上の前回結果を学習前に復元する。
mkdir -p runs
gsutil -m rsync -r "${GCS_BUCKET}/runs/" runs/

echo "=== running: ${TRAIN_CMD} ==="
eval "${TRAIN_CMD}"
TRAIN_EXIT_CODE=$?
echo "=== train command exit code: ${TRAIN_EXIT_CODE} ==="

echo "=== uploading results to ${GCS_BUCKET} ==="
gsutil -m rsync -r runs/ "${GCS_BUCKET}/runs/"

echo "=== done ==="
exit "${TRAIN_EXIT_CODE}"
