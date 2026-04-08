#!/bin/bash

# 修复 numpy 兼容性问题 - 保守方案
# 不重新编译，而是安装预编译的兼容版本

echo "=========================================="
echo "修复 numpy 兼容性问题（保守方案）"
echo "=========================================="

# 激活环境
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"
conda activate /data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed

echo ""
echo "当前版本："
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>&1 || echo "NumPy 未安装或有问题"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" 2>&1 || echo "scikit-learn 未安装或有问题"

echo ""
echo "=========================================="
echo "方案 1: 降级到完全兼容的版本组合"
echo "=========================================="
echo ""
echo "这个组合已知可以正常工作："
echo "  - numpy 1.24.3"
echo "  - scikit-learn 1.3.2"
echo ""

read -p "是否执行方案 1？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装兼容版本..."
    pip install numpy==1.24.3 scikit-learn==1.3.2 --force-reinstall --no-cache-dir

    echo ""
    echo "验证安装..."
    python -c "
import numpy
import sklearn
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ scikit-learn: {sklearn.__version__}')

from sklearn.utils.murmurhash import murmurhash3_32
print('✓ murmurhash OK')

from transformers import Trainer
print('✓ Trainer OK')
"
    exit 0
fi

echo ""
echo "=========================================="
echo "方案 2: 使用 conda 安装（推荐）"
echo "=========================================="
echo ""
echo "conda 会自动处理二进制兼容性"
echo ""

read -p "是否执行方案 2？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "卸载 pip 安装的版本..."
    pip uninstall numpy scikit-learn -y

    echo "使用 conda 安装..."
    conda install numpy scikit-learn -y

    echo ""
    echo "验证安装..."
    python -c "
import numpy
import sklearn
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ scikit-learn: {sklearn.__version__}')

from sklearn.utils.murmurhash import murmurhash3_32
print('✓ murmurhash OK')

from transformers import Trainer
print('✓ Trainer OK')
"
    exit 0
fi

echo ""
echo "=========================================="
echo "方案 3: 完全卸载 scikit-learn"
echo "=========================================="
echo ""
echo "如果 transformers 不需要 scikit-learn，可以直接卸载"
echo ""

read -p "是否执行方案 3？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "卸载 scikit-learn..."
    pip uninstall scikit-learn -y

    echo ""
    echo "验证..."
    python -c "
from transformers import Trainer
print('✓ Trainer OK (without scikit-learn)')
"
    exit 0
fi

echo ""
echo "未选择任何方案，退出。"
