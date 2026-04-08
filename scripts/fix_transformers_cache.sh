#!/usr/bin/env bash

# 修复 transformers 4.36.2 的 EncoderDecoderCache 导入问题

echo "=========================================="
echo "修复 transformers EncoderDecoderCache 问题"
echo "=========================================="

# 激活环境
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"
conda activate /data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed

echo ""
echo "问题分析："
echo "  transformers 4.36.2 不包含 EncoderDecoderCache"
echo "  这个类是在 transformers 4.41+ 才引入的"
echo "  可能是某个依赖或补丁引用了新版本的 API"
echo ""

echo "=========================================="
echo "步骤 1: 完全卸载并重新安装 transformers"
echo "=========================================="

# 完全清理 transformers
pip uninstall transformers -y
pip cache purge

# 重新安装 transformers 4.36.2
echo ""
echo "安装 transformers 4.36.2..."
pip install transformers==4.36.2 --no-cache-dir

echo ""
echo "=========================================="
echo "步骤 2: 重新应用 VILA 补丁"
echo "=========================================="

cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "Site packages: $site_pkg_path"

if [ -d "./vila_u/train/transformers_replace/" ]; then
    echo ""
    echo "应用 VILA transformers 补丁..."
    cp -rv ./vila_u/train/transformers_replace/* $site_pkg_path/transformers/
    echo "✓ 补丁已应用"
else
    echo "⚠️  警告: 找不到 VILA transformers 补丁目录"
fi

echo ""
echo "=========================================="
echo "步骤 3: 检查是否有冲突的依赖"
echo "=========================================="

echo ""
echo "检查可能导致问题的包..."
python -c "
import pkg_resources

# 检查可能引用新版 transformers API 的包
packages_to_check = [
    'sentence-transformers',
    'lmms-eval',
    'accelerate',
]

print('已安装的相关包：')
for pkg in packages_to_check:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  {pkg}: {version}')
    except:
        print(f'  {pkg}: 未安装')
"

echo ""
echo "=========================================="
echo "步骤 4: 验证修复"
echo "=========================================="
echo ""

python -c "
import sys

print('测试 1: transformers 基础导入')
try:
    import transformers
    print(f'  ✓ transformers {transformers.__version__}')
except Exception as e:
    print(f'  ✗ transformers: {e}')
    sys.exit(1)

print('')
print('测试 2: 检查 EncoderDecoderCache')
try:
    from transformers import EncoderDecoderCache
    print('  ⚠️  EncoderDecoderCache 存在（不应该在 4.36.2 中）')
except ImportError:
    print('  ✓ EncoderDecoderCache 不存在（正常）')

print('')
print('测试 3: Trainer 导入')
try:
    from transformers import Trainer
    print('  ✓ Trainer 导入成功')
except Exception as e:
    print(f'  ✗ Trainer 导入失败: {e}')
    print('')
    print('可能的原因：')
    print('  1. VILA 补丁与 transformers 4.36.2 不兼容')
    print('  2. 某个依赖包需要更新版本的 transformers')
    print('')
    print('建议：')
    print('  检查 sentence-transformers 版本，可能需要降级：')
    print('    pip install sentence-transformers==2.2.2')
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
    echo "现在可以开始训练："
    echo "  bash scripts/train/train_action_prediction.sh"
else
    echo ""
    echo "=========================================="
    echo "✗ 仍然失败"
    echo "=========================================="
    echo ""
    echo "可能需要降级 sentence-transformers："
    echo "  pip install sentence-transformers==2.2.2"
    echo ""
    echo "然后重新运行此脚本："
    echo "  bash scripts/fix_transformers_cache.sh"
    exit 1
fi
