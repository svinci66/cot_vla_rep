# 🚨 训练前必读：更新代码

## 问题
你遇到的错误是因为远程服务器上的代码还没有更新到最新版本。

```
ModuleNotFoundError: No module named 'vila_u.train.train_action_prediction_main'
```

## 解决方法

在远程服务器上运行：

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep

# 拉取最新代码
git pull origin main

# 验证文件存在
ls -la vila_u/train/train_action_prediction*.py
ls -la vila_u/data/libero_dataset_v2.py
ls -la scripts/train/train_action_prediction.sh

# 现在可以运行训练了
bash scripts/train/train_action_prediction.sh
```

## 新增的文件

确保以下文件都存在：

1. `vila_u/train/train_action_prediction_mem.py` - 训练入口
2. `vila_u/train/train_action_prediction_main.py` - 主训练逻辑
3. `vila_u/data/libero_dataset_v2.py` - 增强数据加载器
4. `scripts/train/train_action_prediction.sh` - 训练脚本

## 验证

运行以下命令验证文件完整性：

```bash
python -c "
from vila_u.train.train_action_prediction_main import train
from vila_u.data.libero_dataset_v2 import LiberoGoalDataset
print('✓ All modules imported successfully!')
"
```

如果没有报错，说明代码已经更新成功，可以开始训练了。
