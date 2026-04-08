#!/usr/bin/env bash

# 基于 setup_environment.sh 的 numpy 兼容性修复
# 只修复 scikit-learn，不改变其他配置

echo "=========================================="
echo "修复 scikit-learn 的 numpy 兼容性"
echo "=========================================="

# 激活环境
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"
CONDA_ENV_PATH="/data/share/1919650160032350208/sj/conda_pkgs/${1:-"vila_env"}"
conda activate "$CONDA_ENV_PATH"

echo ""
echo "当前环境: $CONDA_ENV_PATH"
echo ""

# 检查当前版本
echo "当前版本："
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "  scikit-learn: 未安装或有问题"

echo ""
echo "=========================================="
echo "方案：使用 conda 安装 scikit-learn"
echo "=========================================="
echo ""
echo "conda 会自动处理二进制兼容性，这是最可靠的方法"
echo ""

# 卸载 pip 安装的 scikit-learn
echo "步骤 1: 卸载 pip 安装的 scikit-learn..."
pip uninstall scikit-learn -y 2>/dev/null

# 使用 conda 安装
echo ""
echo "步骤 2: 使用 conda 安装 scikit-learn..."
conda install scikit-learn -y

echo ""
echo "=========================================="
echo "验证修复"
echo "=========================================="
echo ""

# 验证
python -c "
import sys

# 测试 1: 基础导入
try:
    import numpy
    print(f'✓ NumPy {numpy.__version__}')
except Exception as e:
    print(f'✗ NumPy 导入失败: {e}')
    sys.exit(1)

try:
    import sklearn
    print(f'✓ scikit-learn {sklearn.__version__}')
except Exception as e:
    print(f'✗ scikit-learn 导入失败: {e}')
    sys.exit(1)

# 测试 2: murmurhash (问题的根源)
try:
    from sklearn.utils.murmurhash import murmurhash3_32
    print('✓ sklearn.utils.murmurhash OK')
except Exception as e:
    print(f'✗ murmurhash 导入失败: {e}')
    sys.exit(1)

# 测试 3: transformers.Trainer
try:
    from transformers import Trainer
    print('✓ transformers.Trainer OK')
except Exception as e:
    print(f'✗ Trainer 导入失败: {e}')
    sys.exit(1)

# 测试 4: VILA-U 基础功能
try:
    import vila_u
    print('✓ vila_u OK')
except Exception as e:
    print(f'✗ vila_u 导入失败: {e}')
    sys.exit(1)

# 测试 5: 训练代码
try:
    from vila_u.train.train_action_prediction_main import train
    print('✓ 训练代码导入 OK')
except Exception as e:
    print(f'✗ 训练代码导入失败: {e}')
    sys.exit(1)

print('')
print('🎉 所有测试通过！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 修复成功！"
    echo "=========================================="
    echo ""
    echo "现在可以开始训练："
    echo "  cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep"
    echo "  bash scripts/train/train_action_prediction.sh"
else
    echo ""
    echo "=========================================="
    echo "✗ 修复失败"
    echo "=========================================="
    echo ""
    echo "请尝试完全重建环境："
    echo "  bash setup_environment.sh vila_env_new"
    exit 1
fi
