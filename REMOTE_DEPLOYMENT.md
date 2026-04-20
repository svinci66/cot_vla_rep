# 远程服务器训练部署指南

## 📦 提供的脚本

### 1. `scripts/start_training.sh` - 前台运行
- 直接在当前终端运行
- 可以看到实时输出
- 终端关闭后训练会停止

### 2. `scripts/start_training_background.sh` - 后台运行（推荐）
- 后台运行，不占用终端
- 输出保存到日志文件
- 终端关闭后训练继续

---

## 🚀 快速开始

### 方法1：前台运行（适合测试）

```bash
# 1. 上传脚本到服务器
scp scripts/start_training.sh user@server:/data/private/

# 2. SSH 登录服务器
ssh user@server

# 3. 添加执行权限
chmod +x /data/private/start_training.sh

# 4. 运行
/data/private/start_training.sh
```

### 方法2：后台运行（推荐用于长时间训练）

```bash
# 1. 上传脚本到服务器
scp scripts/start_training_background.sh user@server:/data/private/

# 2. SSH 登录服务器
ssh user@server

# 3. 添加执行权限
chmod +x /data/private/start_training_background.sh

# 4. 后台运行
/data/private/start_training_background.sh

# 5. 查看日志
tail -f /data/private/training_*.log
```

---

## 📊 监控训练

### 查看训练日志
```bash
# 实时查看最新日志
tail -f /data/private/training_*.log

# 查看最近100行
tail -n 100 /data/private/training_*.log

# 搜索特定内容
grep "loss" /data/private/training_*.log
```

### 查看GPU使用情况
```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看特定GPU
nvidia-smi -i 0,1,2,3,4,5,6,7
```

### 查看训练进程
```bash
# 查看进程是否在运行
ps aux | grep train_action_prediction

# 使用保存的PID
cat /data/private/training.pid
ps -p $(cat /data/private/training.pid)
```

---

## 🛑 停止训练

### 方法1：使用保存的PID
```bash
# 优雅停止
kill $(cat /data/private/training.pid)

# 强制停止（如果上面不行）
kill -9 $(cat /data/private/training.pid)
```

### 方法2：查找并停止
```bash
# 查找进程
ps aux | grep train_action_prediction

# 停止进程（替换 <PID> 为实际的进程ID）
kill <PID>
```

### 方法3：停止所有训练进程
```bash
# 谨慎使用！会停止所有训练
pkill -f train_action_prediction
```

---

## 🔧 自定义配置

### 修改GPU数量
编辑脚本中的这些行：
```bash
export NUM_GPUS=4  # 改为4卡
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 只使用前4张卡
export BATCH_SIZE=80  # 相应调整batch size
```

### 修改其他参数
在脚本的"启动训练任务"部分添加：
```bash
export LEARNING_RATE=2e-5
export NUM_EPOCHS=10
export USE_TORCH_COMPILE=True
```

### 使用不同的训练脚本
```bash
# 如果要用 L40 配置
bash scripts/train/train_action_prediction_l40.sh
```

---

## 📝 脚本说明

### start_training.sh 做了什么？

1. ✅ 加载 Conda 配置
2. ✅ 激活 vila_env_fixed 环境
3. ✅ 进入项目目录
4. ✅ 拉取最新代码
5. ✅ 启动8卡训练

### start_training_background.sh 额外功能

6. ✅ 后台运行（nohup）
7. ✅ 输出重定向到日志文件
8. ✅ 保存进程PID
9. ✅ 提供监控命令

---

## ⚠️ 注意事项

### 1. 路径检查
确保这些路径存在：
- `/data/private/miniconda.sh` - Conda 配置
- `/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed` - Conda 环境
- `/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep` - 项目目录

### 2. 权限检查
```bash
# 确保脚本有执行权限
chmod +x /data/private/start_training*.sh

# 确保日志目录可写
touch /data/private/test.log
rm /data/private/test.log
```

### 3. GPU可用性
```bash
# 检查GPU是否可用
nvidia-smi

# 检查是否有其他进程占用GPU
nvidia-smi | grep python
```

### 4. 磁盘空间
```bash
# 检查磁盘空间（训练会生成checkpoint）
df -h /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep/checkpoints
```

---

## 🐛 故障排查

### 问题1：Conda 环境激活失败
```bash
# 手动测试
source /data/private/miniconda.sh
conda activate /data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed
which python
```

### 问题2：找不到训练脚本
```bash
# 检查项目目录
ls -la /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep/scripts/train/

# 确保在正确的分支
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep
git branch
```

### 问题3：训练立即退出
```bash
# 查看日志找出原因
cat /data/private/training_*.log

# 常见原因：
# - 显存不足 → 减小 BATCH_SIZE
# - 数据路径错误 → 检查 DATA_ROOT
# - 模型路径错误 → 检查 MODEL_PATH
```

### 问题4：无法后台运行
```bash
# 确保使用了 nohup
nohup /data/private/start_training.sh > /data/private/training.log 2>&1 &

# 或者使用 screen/tmux
screen -S training
/data/private/start_training.sh
# Ctrl+A, D 分离
```

---

## 📈 预期输出

### 启动成功的标志
```
>>> Sourcing miniconda.sh...
>>> Activating conda environment: ...
>>> Conda environment activated: /data/share/.../bin/python
>>> Current directory: /data/share/.../cot_vla_rep
>>> Already up to date.
>>> Starting training...
>>> Config: 8 GPUs, Batch Size 160
>>> Training started with PID: 12345
```

### 训练日志示例
```
Loading checkpoint shards: 100%
Model loaded successfully
[LiberoGoalDataset] Loaded 58598 samples
================================================================================
Training Configuration
================================================================================
  Single GPU Mode: False
  GPUs per Node: 8
  Batch Size: 160
  ...
================================================================================
Training started...
{'loss': 2.345, 'learning_rate': 1e-05, 'epoch': 0.01}
{'loss': 2.123, 'learning_rate': 1.2e-05, 'epoch': 0.02}
...
```

---

## 🎯 完整工作流程

```bash
# 1. 上传脚本
scp scripts/start_training_background.sh user@server:/data/private/

# 2. 登录服务器
ssh user@server

# 3. 添加执行权限
chmod +x /data/private/start_training_background.sh

# 4. 启动训练
/data/private/start_training_background.sh

# 5. 查看日志
tail -f /data/private/training_*.log

# 6. 监控GPU（另开终端）
watch -n 1 nvidia-smi

# 7. 等待训练完成...

# 8. 检查checkpoint
ls -lh /data/share/.../cot_vla_rep/checkpoints/
```

---

## 💡 高级用法

### 使用 tmux（推荐）
```bash
# 创建新会话
tmux new -s training

# 运行训练
/data/private/start_training.sh

# 分离会话：Ctrl+B, D

# 重新连接
tmux attach -t training

# 查看所有会话
tmux ls
```

### 使用 screen
```bash
# 创建新会话
screen -S training

# 运行训练
/data/private/start_training.sh

# 分离会话：Ctrl+A, D

# 重新连接
screen -r training

# 查看所有会话
screen -ls
```

### 定时启动
```bash
# 使用 crontab 定时启动
crontab -e

# 添加：每天凌晨2点启动训练
0 2 * * * /data/private/start_training_background.sh
```

---

## 📞 需要帮助？

如果遇到问题：
1. 查看日志文件
2. 检查GPU使用情况
3. 确认所有路径正确
4. 查看本文档的故障排查部分
