#!/bin/bash

# 克隆现有 Conda 环境并修复 numpy 兼容性问题

echo "=========================================="
echo "克隆 Conda 环境并修复依赖"
echo "=========================================="

# 配置
SOURCE_ENV="/data/share/1919650160032350208/sj/conda_pkgs/vila_env"
NEW_ENV="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed"

echo ""
echo "源环境: $SOURCE_ENV"
echo "新环境: $NEW_ENV"
echo ""

# 激活 Conda
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"

# 检查源环境是否存在
if [ ! -d "$SOURCE_ENV" ]; then
    echo "✗ 错误: 源环境不存在: $SOURCE_ENV"
    exit 1
fi

# 克隆环境
echo "步骤 1: 克隆环境 (这可能需要几分钟)..."
if [ -d "$NEW_ENV" ]; then
    echo "⚠ 新环境已存在，将被删除..."
    conda remove -p "$NEW_ENV" --all -y
fi

conda create -p "$NEW_ENV" --clone "$SOURCE_ENV" -y

if [ $? -ne 0 ]; then
    echo "✗ 克隆失败"
    exit 1
fi

echo "✓ 环境克隆成功"
echo ""

# 激活新环境
echo "步骤 2: 激活新环境..."
conda activate "$NEW_ENV"

# 验证激活
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $NF}')
echo "当前环境: $CURRENT_ENV"
echo ""

# 修复 scikit-learn
echo "步骤 3: 重新编译 scikit-learn..."
pip install --upgrade --force-reinstall --no-cache-dir scikit-learn

if [ $? -ne 0 ]; then
    echo "✗ scikit-learn 重新安装失败"
    exit 1
fi

echo "✓ scikit-learn 修复完成"
echo ""

# 验证修复
echo "=========================================="
echo "验证修复结果"
echo "=========================================="

echo ""
echo "测试 1: sklearn.utils.murmurhash"
python -c "from sklearn.utils.murmurhash import murmurhash3_32; print('✓ sklearn OK')" 2>&1

echo ""
echo "测试 2: transformers.Trainer"
python -c "from transformers import Trainer; print('✓ Trainer OK')" 2>&1

echo ""
echo "测试 3: 训练代码导入"
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep
python -c "from vila_u.train.train_action_prediction_main import train; print('✓ 训练代码 OK')" 2>&1

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
echo ""
echo "新环境路径: $NEW_ENV"
echo ""
echo "使用新环境:"
echo "  conda activate $NEW_ENV"
echo ""
echo "或者在脚本中使用:"
echo "  export CONDA_ENV_PATH=$NEW_ENV"
echo "  bash scripts/train/train_action_prediction.sh"
echo ""
echo "如果一切正常，可以删除旧环境:"
echo "  conda remove -p $SOURCE_ENV --all"
