"""
检查远程服务器上的代码版本
"""

import sys
import os

# 检查文件内容
test_file = '/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep/tests/test_model_loading.py'

print("Checking test_model_loading.py on remote server...")
print("=" * 70)

with open(test_file, 'r') as f:
    lines = f.readlines()

# 检查关键行
print(f"\nLine 103-120:")
for i in range(102, min(120, len(lines))):
    print(f"{i+1:3d}: {lines[i]}", end='')

print("\n" + "=" * 70)

# 检查是否包含 predict_action
if 'predict_action' in ''.join(lines[100:130]):
    print("✓ Code contains predict_action method")
else:
    print("✗ Code does NOT contain predict_action method")
    print("  Remote server needs to run: git pull")

# 检查 git 状态
print("\n" + "=" * 70)
print("Git status:")
os.system('cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep && git log --oneline -3')
