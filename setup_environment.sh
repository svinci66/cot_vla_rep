#!/usr/bin/env bash

# 注意：如果是 source 执行，建议暂时关闭 set -e 以防断连
# set -e 

# 1. 定义公共目录下的环境路径
PUBLIC_BASE="/data/share/1919650160032350208/sj/conda_pkgs" 
CONDA_ENV_PATH="$PUBLIC_BASE/${1:-"vila_env"}"

# 2. 激活 Conda 钩子 (指向公共目录的 miniconda)
eval "$(/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda shell.bash hook)"

echo "=========================================================="
echo "1. 在公共目录创建/激活环境: $CONDA_ENV_PATH"
echo "=========================================================="
if [ ! -d "$CONDA_ENV_PATH" ]; then
    conda create -p "$CONDA_ENV_PATH" python=3.12 -y
fi
conda activate "$CONDA_ENV_PATH"

# 3. 设置 Pip 缓存到公共目录（防止重复下载大型包）
export PIP_CACHE_DIR=/data/share/1919650160032350208/sj/pip_cache

echo "=========================================================="
echo "2. 更新基础打包工具"
echo "=========================================================="
pip install --upgrade pip setuptools wheel

echo "=========================================================="
echo "3. 安装 PyTorch 2.8.0 (匹配 RTX 5090 等高算力架构)"
echo "=========================================================="
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=========================================================="
echo "4. 安装 Flash-Attention 2.8.3 (本地免编译)"
echo "=========================================================="
pip install /data/share/1919650160032350208/sj/conda_env/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

echo "=========================================================="
echo "5. 自动解除 VILA 源码的版本死锁 (关键防护)"
echo "=========================================================="
if [ -f "pyproject.toml" ]; then
    echo "检测到 pyproject.toml，正在注入依赖放宽补丁..."
    # 【防止降级】保护 PyTorch 及底层加速库不被拉回不支持 5090 的旧版本
    sed -i 's/"torch==[0-9\.]*"/"torch>=2.8.0"/g' pyproject.toml
    sed -i 's/"torchvision==[0-9\.]*"/"torchvision>=0.19.0"/g' pyproject.toml
    sed -i 's/"accelerate==[0-9\.]*"/"accelerate>=0.34.2"/g' pyproject.toml
    sed -i 's/"bitsandbytes==[0-9\.]*"/"bitsandbytes>=0.41.0"/g' pyproject.toml
    sed -i 's/"numpy==[0-9\.]*"/"numpy>=1.26.4"/g' pyproject.toml
    
    # 【防止编译失败】强制使用新版 sentencepiece，绕过老版本缺少 cmake 的报错
    sed -i 's/"sentencepiece==0.1.99"/"sentencepiece>=0.1.99"/g' pyproject.toml
    
    # 【防止评测包冲突】放宽 datasets 版本限制，解决与 lmms-eval 的冲突
    sed -i 's/"datasets==2.16.1"/"datasets>=2.19.0"/g' pyproject.toml
fi

echo "=========================================================="
echo "6. 安装 VILA 主程序及评测依赖"
echo "=========================================================="
pip install -e ".[train,eval]"
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

echo "=========================================================="
echo "7. 运行时异常排雷包 (针对 Python 3.12 和纯净服务器)"
echo "=========================================================="
# 7.1 【保全核心配置】强制统一 antlr4 版本，解决 Hydra-core 和 Latex 解析包的冲突
pip install antlr4-python3-runtime==4.9.3

# 7.2 【解决图形库缺失】卸载可能存在的旧 cv2，安装为服务器打造的无头版，避免 libGL.so.1 报错
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless

# 7.3 【解决循环导入】强制降级 transformers 以严格匹配 VILA 的魔改补丁
pip install transformers==4.36.2

# 7.4 【修复 numpy 兼容性】重新编译 scikit-learn 以匹配当前 numpy 版本
echo "重新编译 scikit-learn 以修复 numpy dtype 兼容性问题..."
# 先锁定 numpy 版本到 1.x
pip install "numpy>=1.26.4,<2.0" --force-reinstall --no-cache-dir
# 重新编译 scikit-learn，不允许升级依赖
pip install --upgrade --force-reinstall --no-cache-dir --no-deps scikit-learn
# 安装 scikit-learn 的依赖（但不升级已有的包）
pip install joblib scipy threadpoolctl --no-upgrade

# 7.5 【解决语法红错】修复 Python 3.12 dataclasses 中可变默认值的语法校验失败
CONFIG_FILE="vila_u/model/multimodal_encoder/rqvaesigliptransformer/rqtransformer/configuration_rqtransformer.py"
if [ -f "$CONFIG_FILE" ]; then
    # 利用 sed 注入 field 模块并替换掉错误的类实例直接赋值
    sed -i 's/from dataclasses import dataclass/from dataclasses import dataclass, field/g' "$CONFIG_FILE"
    sed -i 's/block: AttentionBlockConfig = AttentionBlockConfig()/block: AttentionBlockConfig = field(default_factory=AttentionBlockConfig)/g' "$CONFIG_FILE"
fi

echo "=========================================================="
echo "8. 应用 VILA 的 Transformers 魔改补丁"
echo "=========================================================="
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
if [ -d "./vila_u/train/transformers_replace/" ]; then
    echo "正在将定制补丁复制到: $site_pkg_path/transformers/"
    cp -rv ./vila_u/train/transformers_replace/* $site_pkg_path/transformers/
fi

# 清除可能导致冲突的评测模型文件
rm -rf $site_pkg_path/lmms_eval/models/mplug_owl_video/modeling_mplug_owl.py

echo "=========================================================="
echo "🎉 环境配置全部完成！你的模型现在可以开始狂奔了！"
echo "=========================================================="