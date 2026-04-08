#!/usr/bin/env bash

# H20 专用：修复 transformers 4.36.2 在 Python 3.12 环境下的兼容性问题
# 只在 vila_env_fixed 环境下操作，不影响原始环境

echo "=========================================="
echo "H20 环境修复脚本（仅修改 vila_env_fixed）"
echo "=========================================="

# 激活环境
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"

# 强制使用 vila_env_fixed，忽略参数
CONDA_ENV_PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed"

# 检查环境是否存在
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo ""
    echo "✗ 错误: vila_env_fixed 环境不存在"
    echo ""
    echo "请先运行克隆脚本创建环境："
    echo "  bash scripts/clone_and_fix_env.sh"
    exit 1
fi

conda activate "$CONDA_ENV_PATH"

echo ""
echo "✓ 已激活环境: $CONDA_ENV_PATH"
echo "✓ 原始环境 vila_env 不会被修改"
echo "Python 版本: $(python --version)"
echo ""

# 检查当前状态
echo "=========================================="
echo "诊断当前环境"
echo "=========================================="
echo ""

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import numpy
    print(f'NumPy: {numpy.__version__}')
except:
    print('NumPy: 未安装')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.version.cuda}')
except:
    print('PyTorch: 未安装')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except:
    print('Transformers: 未安装')

try:
    import sklearn
    print(f'scikit-learn: {sklearn.__version__}')
except:
    print('scikit-learn: 未安装')
"

echo ""
echo "=========================================="
echo "修复方案：使用 conda 管理关键依赖"
echo "=========================================="
echo ""

# 方案：使用 conda 安装 scikit-learn 和相关依赖
echo "步骤 1: 卸载 pip 安装的 scikit-learn..."
pip uninstall scikit-learn -y 2>/dev/null

echo ""
echo "步骤 2: 使用 conda 安装 scikit-learn（自动处理二进制兼容性）..."
conda install scikit-learn -c conda-forge -y

echo ""
echo "步骤 3: 确保 numpy 版本正确..."
# 检查 numpy 版本，如果是 2.x 则降级
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [[ $NUMPY_VERSION == 2.* ]]; then
    echo "检测到 NumPy 2.x，降级到 1.x..."
    conda install "numpy>=1.26,<2.0" -y
else
    echo "NumPy 版本正常: $NUMPY_VERSION"
fi

echo ""
echo "步骤 4: 重新应用 VILA transformers 补丁..."
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
if [ -d "./vila_u/train/transformers_replace/" ]; then
    echo "正在将定制补丁复制到: $site_pkg_path/transformers/"
    cp -rv ./vila_u/train/transformers_replace/* $site_pkg_path/transformers/
fi

echo ""
echo "=========================================="
echo "验证修复"
echo "=========================================="
echo ""

python -c "
import sys

print('测试 1: 基础库导入')
try:
    import numpy
    print(f'  ✓ NumPy {numpy.__version__}')
except Exception as e:
    print(f'  ✗ NumPy: {e}')
    sys.exit(1)

try:
    import torch
    print(f'  ✓ PyTorch {torch.__version__}')
except Exception as e:
    print(f'  ✗ PyTorch: {e}')
    sys.exit(1)

print('')
print('测试 2: scikit-learn')
try:
    import sklearn
    print(f'  ✓ scikit-learn {sklearn.__version__}')
except Exception as e:
    print(f'  ✗ scikit-learn: {e}')
    sys.exit(1)

try:
    from sklearn.utils.murmurhash import murmurhash3_32
    print('  ✓ murmurhash OK')
except Exception as e:
    print(f'  ✗ murmurhash: {e}')
    sys.exit(1)

print('')
print('测试 3: transformers')
try:
    import transformers
    print(f'  ✓ transformers {transformers.__version__}')
except Exception as e:
    print(f'  ✗ transformers: {e}')
    sys.exit(1)

try:
    from transformers import Trainer
    print('  ✓ Trainer OK')
except Exception as e:
    print(f'  ✗ Trainer: {e}')
    print('')
    print('提示：如果这里失败，可能需要重新安装 transformers')
    print('  pip uninstall transformers -y')
    print('  pip install transformers==4.36.2')
    sys.exit(1)

print('')
print('测试 4: VILA-U')
try:
    import vila_u
    print('  ✓ vila_u OK')
except Exception as e:
    print(f'  ✗ vila_u: {e}')
    sys.exit(1)

print('')
print('测试 5: 训练代码')
try:
    from vila_u.train.train_action_prediction_main import train
    print('  ✓ 训练代码导入 OK')
except Exception as e:
    print(f'  ✗ 训练代码: {e}')
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
    echo "环境信息："
    python -c "
import numpy, torch, transformers, sklearn
print(f'  Python: 3.12')
print(f'  PyTorch: {torch.__version__} (CUDA {torch.version.cuda})')
print(f'  NumPy: {numpy.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
"
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
    echo "建议："
    echo "1. 检查上面的错误信息"
    echo "2. 如果是 transformers.Trainer 失败，尝试："
    echo "     pip uninstall transformers -y"
    echo "     pip install transformers==4.36.2"
    echo "     bash $0"
    echo ""
    echo "3. 如果问题持续，考虑重建环境："
    echo "     bash setup_environment.sh vila_env_new"
    exit 1
fi
