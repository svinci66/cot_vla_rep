CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VI_START_TOKEN = "<vi_start>"
DEFAULT_VI_END_TOKEN = "<vi_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

SENTINEL_TOKEN = "<vila/sentinel>"

# ===== Action Prediction (方案A - 无CoT) =====
ACTION_DIM = 7              # 7-DoF 动作: [dx, dy, dz, droll, dpitch, dyaw, gripper]
ACTION_CHUNK_SIZE = 10      # 每次预测的动作步数
ACTION_HORIZON = 10         # 动作预测的时间跨度
ACTION_NUM_BINS = 256       # Phase 2: 每个动作维度离散化 bin 数
ACTION_MIN = -1.0           # 动作归一化最小值
ACTION_MAX = 1.0            # 动作归一化最大值

# ===== Phase 4: Visual CoT =====
ACTION_TOKEN_INDEX = -201
DEFAULT_ACT_START_TOKEN = "<act>"
DEFAULT_ACT_END_TOKEN = "</act>"
DEFAULT_SUBGOAL_START_TOKEN = "<subgoal>"
DEFAULT_SUBGOAL_END_TOKEN = "</subgoal>"

SUBGOAL_HORIZON_LOW = 4     # 子目标采样下界（帧数）
SUBGOAL_HORIZON_HIGH = 16   # 子目标采样上界（帧数）
