# LIBERO Goal Dataset Reader - 安装和使用说明

## 文件说明

在 `vila-u-main` 目录下创建了以下文件：

1. **libero_goal_reader.py** (11KB)
   - 核心读取器类 `LiberoGoalReader`
   - 支持读取 LIBERO Goal 的 HDF5 数据集
   - 提供数据集信息查询、加载、保存功能

2. **test_libero_reader.py** (3.5KB)
   - 测试脚本，演示如何使用 LiberoGoalReader
   - 包含6个测试用例

3. **libero_usage_examples.py** (5.2KB)
   - 详细的使用示例和API说明
   - 数据集结构说明
   - 与VILA-U集成的建议

4. **README_LIBERO.md** (3KB)
   - 快速入门指南
   - 命令行使用说明
   - 数据集结构文档

5. **requirements_libero.txt** (65B)
   - 依赖包列表：h5py, numpy

## 安装依赖

```bash
pip install h5py numpy
# 或者
pip install -r requirements_libero.txt
```

## 快速使用

### 1. Python API

```python
from libero_goal_reader import LiberoGoalReader

# 读取数据集
reader = LiberoGoalReader("path/to/libero_goal/task_demo.hdf5")

# 获取数据集信息
info = reader.get_dataset_info(verbose=True)

# 加载完整数据集
dataset = reader.load_dataset()

# 访问数据
for demo in dataset['demonstrations']:
    actions = demo['actions']  # (T, 7)
    images = demo['obs']['agentview_rgb']  # (T, H, W, 3)

# 保存数据集
reader.save_dataset(dataset, "output.hdf5")
```

### 2. 命令行使用

```bash
# 查看数据集信息
python libero_goal_reader.py --input dataset.hdf5 --info

# 读取并保存
python libero_goal_reader.py --input dataset.hdf5 --output new.hdf5

# 读取特定演示
python libero_goal_reader.py --input dataset.hdf5 --demo-indices 0 1 2 --output subset.hdf5
```

## 数据集结构

LIBERO Goal 数据集包含：
- **语言指令**: 描述任务目标的自然语言
- **演示轨迹**: 多个完整的任务执行轨迹
- **观察数据**: RGB图像、深度图、机器人状态等
- **动作序列**: 7维动作 (3位置 + 3旋转 + 1夹爪)
- **奖励和终止标志**

## 需要说明的问题

1. **数据集路径**: 需要提供实际的 LIBERO Goal 数据集文件路径（.hdf5格式）
2. **依赖安装**: 当前环境缺少 h5py，需要先安装依赖
3. **数据格式转换**: 如果VILA-U需要特定的数据格式，可能需要额外的转换逻辑

## 下一步

如果你有具体的LIBERO Goal数据集文件，可以：
1. 安装依赖：`pip install h5py numpy`
2. 运行测试：`python test_libero_reader.py /path/to/dataset.hdf5`
3. 查看示例：`python libero_usage_examples.py`

如果需要与VILA-U的特定格式集成，请告诉我VILA-U期望的数据格式，我可以添加相应的转换功能。
