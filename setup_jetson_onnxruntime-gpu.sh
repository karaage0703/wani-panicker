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

# JetPack 6.0 用 ONNX Runtime GPU のダウンロードURL (Python 3.10)
declare -A ONNX_URLS
ONNX_URLS["3.10"]="https://nvidia.box.com/shared/static/6l0u97rj80ifwkk8rqbzj1try89fk26z.whl"

# 対応するPythonバージョンかチェック
if [[ ! ${ONNX_URLS[$PYTHON_VERSION]} ]]; then
    echo "❌ サポートされていないPythonバージョン: $PYTHON_VERSION"
    echo "   サポート版: 3.8, 3.9, 3.10, 3.11, 3.12"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION はサポートされています"

# ONNX Runtime GPU のダウンロードとインストール
echo "🔥 ONNX Runtime GPU (JetPack 6.0 対応) をダウンロード中..."
ONNX_URL=${ONNX_URLS[$PYTHON_VERSION]}
ONNX_WHEEL="onnxruntime_gpu-1.18.0-cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}-linux_aarch64.whl"

# 既存のONNX Runtimeをアンインストール
echo "🗑️ 既存のONNX Runtimeをアンインストール中..."
pip3 uninstall -y onnxruntime onnxruntime-gpu || true

# ダウンロードとインストール
wget -O "$ONNX_WHEEL" "$ONNX_URL"
pip3 install "$ONNX_WHEEL"

# ONNX Runtime の動作確認
echo "✅ ONNX Runtime 動作確認中..."
python3 -c "
import onnxruntime as ort
print('ONNX Runtime バージョン:', ort.__version__)
print('利用可能なプロバイダー:', ort.get_available_providers())

# TensorRT と CUDA プロバイダーが利用可能かチェック
providers = ort.get_available_providers()
if 'TensorrtExecutionProvider' in providers:
    print('✅ TensorRT プロバイダー: 利用可能')
else:
    print('⚠️  TensorRT プロバイダー: 利用不可')

if 'CUDAExecutionProvider' in providers:
    print('✅ CUDA プロバイダー: 利用可能')
else:
    print('⚠️  CUDA プロバイダー: 利用不可')
"

# クリーンアップ
echo "🧹 ダウンロードファイルをクリーンアップ中..."
rm -f "$ONNX_WHEEL"

# 最終確認
echo ""
echo "🎉 セットアップ完了!"
echo "================================="
