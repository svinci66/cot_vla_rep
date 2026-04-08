#!/usr/bin/env bash

# 修复 PEFT 版本兼容性问题

echo "=========================================="
echo "修复 PEFT 与 transformers 4.36.2 的兼容性"
echo "=========================================="

# 激活环境
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"
conda activate /data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed

echo ""
echo "问题分析："
echo "  PEFT 新版本需要 transformers 4.41+"
echo "  但 VILA 需要 transformers 4.36.2"
echo "  解决方案：降级 PEFT 到兼容版本"
echo ""

echo "=========================================="
echo "步骤 1: 检查当前 PEFT 版本"
echo "=========================================="
echo ""

python -c "
try:
    import peft
    print(f'当前 PEFT 版本: {peft.__version__}')
except:
    print('PEFT 未安装')
"

echo ""
echo "=========================================="
echo "步骤 2: 降级 PEFT 到 0.7.1"
echo "=========================================="
echo ""
echo "PEFT 0.7.1 是最后一个支持 transformers 4.36.x 的版本"
echo ""

pip install peft==0.7.1 --no-deps

echo ""
echo "=========================================="
echo "步骤 3: 验证修复"
echo "=========================================="
echo ""

python -c "
import sys

print('测试 1: PEFT 版本')
try:
    import peft
    print(f'  ✓ PEFT {peft.__version__}')
except Exception as e:
    print(f'  ✗ PEFT: {e}')
    sys.exit(1)

print('')
print('测试 2: transformers 版本')
try:
    import transformers
    print(f'  ✓ transformers {transformers.__version__}')
except Exception as e:
    print(f'  ✗ transformers: {e}')
    sys.exit(1)

print('')
print('测试 3: Trainer 导入')
try:
    from transformers import Trainer
    print('  ✓ Trainer 导入成功')
except Exception as e:
    print(f'  ✗ Trainer 导入失败: {e}')
    import traceback
    traceback.print_exc()
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
    echo "版本信息："
    python -c "
import peft, transformers
print(f'  PEFT: {peft.__version__}')
print(f'  transformers: {transformers.__version__}')
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
    echo "如果问题持续，可能需要完全卸载 PEFT："
    echo "  pip uninstall peft -y"
    echo ""
    echo "然后重新安装："
    echo "  pip install peft==0.7.1"
    exit 1
fi
