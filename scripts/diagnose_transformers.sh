#!/usr/bin/env bash

# 深度诊断 transformers.Trainer 导入失败的原因

echo "=========================================="
echo "深度诊断 transformers 导入问题"
echo "=========================================="

# 激活环境
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"
conda activate /data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed

echo ""
echo "环境: vila_env_fixed"
echo ""

echo "=========================================="
echo "步骤 1: 检查 transformers 安装"
echo "=========================================="
echo ""

python -c "
import transformers
import os

print(f'transformers 版本: {transformers.__version__}')
print(f'transformers 路径: {transformers.__file__}')
print(f'transformers 目录: {os.path.dirname(transformers.__file__)}')
"

echo ""
echo "=========================================="
echo "步骤 2: 检查 __init__.py 中的导出"
echo "=========================================="
echo ""

python -c "
import transformers
import inspect

# 检查 transformers 模块中是否有 EncoderDecoderCache
if hasattr(transformers, 'EncoderDecoderCache'):
    print('⚠️  EncoderDecoderCache 存在于 transformers 模块中')
    print(f'   来源: {inspect.getfile(transformers.EncoderDecoderCache)}')
else:
    print('✓ EncoderDecoderCache 不在 transformers 模块中（正常）')

# 检查 __all__ 列表
if hasattr(transformers, '__all__'):
    if 'EncoderDecoderCache' in transformers.__all__:
        print('⚠️  EncoderDecoderCache 在 __all__ 中')
    else:
        print('✓ EncoderDecoderCache 不在 __all__ 中（正常）')
"

echo ""
echo "=========================================="
echo "步骤 3: 尝试导入 Trainer 并捕获详细错误"
echo "=========================================="
echo ""

python -c "
import sys
import traceback

try:
    from transformers import Trainer
    print('✓ Trainer 导入成功')
except Exception as e:
    print('✗ Trainer 导入失败')
    print('')
    print('完整错误堆栈：')
    traceback.print_exc()
    print('')

    # 尝试找出是哪个文件引用了 EncoderDecoderCache
    print('分析错误...')
    tb = sys.exc_info()[2]
    while tb is not None:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename
        lineno = tb.tb_lineno
        if 'transformers' in filename:
            print(f'  文件: {filename}')
            print(f'  行号: {lineno}')
        tb = tb.tb_next
"

echo ""
echo "=========================================="
echo "步骤 4: 检查可能的冲突包"
echo "=========================================="
echo ""

python -c "
import pkg_resources

packages = [
    'transformers',
    'sentence-transformers',
    'lmms-eval',
    'accelerate',
    'datasets',
]

print('已安装包版本：')
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  {pkg}: {version}')
    except:
        print(f'  {pkg}: 未安装')
"

echo ""
echo "=========================================="
echo "步骤 5: 检查 VILA 补丁是否正确应用"
echo "=========================================="
echo ""

site_pkg=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "Site packages: $site_pkg"
echo ""

# 检查关键文件
echo "检查 VILA 补丁文件："
if [ -f "$site_pkg/transformers/models/llama/modeling_llama.py" ]; then
    echo "  ✓ modeling_llama.py 存在"
    # 检查是否是 VILA 版本
    if grep -q "VILA" "$site_pkg/transformers/models/llama/modeling_llama.py" 2>/dev/null; then
        echo "    ✓ 包含 VILA 标记"
    else
        echo "    ⚠️  可能不是 VILA 版本"
    fi
else
    echo "  ✗ modeling_llama.py 不存在"
fi

if [ -f "$site_pkg/transformers/generation/utils.py" ]; then
    echo "  ✓ generation/utils.py 存在"
else
    echo "  ✗ generation/utils.py 不存在"
fi

echo ""
echo "=========================================="
echo "建议"
echo "=========================================="
echo ""
echo "请将上面的完整输出发给我，特别是："
echo "  1. 步骤 3 的完整错误堆栈"
echo "  2. 步骤 4 的包版本列表"
echo "  3. 步骤 5 的补丁检查结果"
echo ""
echo "这样我可以准确定位问题所在。"
