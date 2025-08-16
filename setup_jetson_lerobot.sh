#!/bin/bash

# Jetson Orin用セットアップスクリプト
# Wani Panicker プロジェクト用

set -e

echo "🚀 Jetson Orin セットアップ開始"
echo "================================="

# システム情報確認
echo "📋 システム情報確認中..."
cat /etc/nv_tegra_release
echo ""

# Python バージョン確認
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python バージョン: $PYTHON_VERSION"

#sudo apt remove python3-pandas python3-numpy
pip3 install lerobot
pip3 install "lerobot[feetech]"

#pip3 uninstall -y opencv-python-headless