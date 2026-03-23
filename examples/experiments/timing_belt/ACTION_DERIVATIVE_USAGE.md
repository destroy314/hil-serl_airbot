# Action Derivative Constraint Wrapper - Usage Guide

这个wrapper用于限制策略输出的action的前N阶导数（位置、速度、加速度、加加速度），确保机器人运动平滑且符合物理规律。

## 工作原理

Wrapper维护action的历史记录，并计算：
- **速度 (1阶导数)**: `action[t] - action[t-1]`
- **加速度 (2阶导数)**: `velocity[t] - velocity[t-1]`
- **加加速度 (3阶导数)**: `acceleration[t] - acceleration[t-1]`

当策略输出的action导数超出限制时，wrapper会将其剪裁到允许的范围内。

## 使用步骤

### 1. 从人类演示中提取导数统计信息

首先从录制的演示数据中计算导数的最大值：

```bash
cd examples

# 从单个或多个演示文件中提取统计信息
python compute_action_derivative_stats.py \
    --demo_path=./demo_data/timing_belt_20_demos_2024-01-01_12-00-00.pkl \
    --demo_path=./demo_data/timing_belt_20_demos_2024-01-02_12-00-00.pkl \
    --output_path=./experiments/timing_belt/action_derivative_stats.json
```

这会生成一个JSON文件，包含：
- `max_velocity`: 每个维度的最大速度
- `max_acceleration`: 每个维度的最大加速度
- `max_jerk`: 每个维度的最大加加速度
- 额外包含95th和99th百分位数，供参考

输出示例：
```
================================================================================
VELOCITY (1st derivative: action[t] - action[t-1])
================================================================================
   Dim         Max   95th %ile   99th %ile
     0    0.045123    0.038456    0.042789
     1    0.052341    0.043210    0.049876
     ...

================================================================================
ACCELERATION (2nd derivative)
================================================================================
   Dim         Max   95th %ile   99th %ile
     0    0.012345    0.009876    0.011234
     ...
```

### 2. 在config.py中配置wrapper

编辑 `examples/experiments/timing_belt/config.py`：

```python
class TrainConfig(DefaultTrainingConfig):
    # ... 其他配置 ...

    # Action derivative constraint settings
    action_derivative_stats_path = "./experiments/timing_belt/action_derivative_stats.json"
    action_derivative_scale = 1.0  # 缩放因子，见下面说明
```

**scale参数说明：**
- `scale = 1.0`: 使用演示中的精确限制值（推荐初始值）
- `scale = 1.2`: 允许策略比演示更自由20%
- `scale = 0.8`: 比演示更严格20%（更保守）

### 3. 训练模型

正常运行训练脚本，wrapper会自动应用：

```bash
cd examples/experiments/timing_belt
bash run_learner.sh  # 在GPU机器上
bash run_actor.sh    # 在机器人控制机器上
```

训练过程中，wrapper会：
- 在constraint被触发时，在info中添加 `action_constrained=True`
- 记录 `action_constraint_diff` 显示约束程度

### 4. 监控约束效果

如果想检查约束是否被频繁触发，可以在训练循环中打印：

```python
obs, reward, done, truncated, info = env.step(action)
if info.get("action_constrained", False):
    print(f"Action constrained, diff: {info['action_constraint_diff']:.4f}")
```

## 高级用法

### 手动指定限制（不使用演示统计）

```python
from airbot_env.envs.wrappers import ActionDerivativeConstraintWrapper
import numpy as np

# 手动指定每个维度的限制
max_velocity = np.array([0.05, 0.05, 0.05, ...])  # 14维
max_acceleration = np.array([0.01, 0.01, 0.01, ...])
max_jerk = np.array([0.005, 0.005, 0.005, ...])

env = ActionDerivativeConstraintWrapper(
    env,
    max_velocity=max_velocity,
    max_acceleration=max_acceleration,
    max_jerk=max_jerk,
    scale=1.0
)
```

### 不同任务使用不同的scale

```python
class TrainConfig(DefaultTrainingConfig):
    # 录制演示时使用scale=1.0
    # 在线训练时逐渐放宽约束
    action_derivative_scale = 1.0  # 初始训练
    # action_derivative_scale = 1.2  # 后期fine-tuning
```

## Wrapper在环境栈中的位置

**重要**: ActionDerivativeConstraintWrapper应该在以下wrapper之后：
1. DualAirbotIntervention (人类干预)
2. DeltaJointActionWrapper (如果使用相对控制)

在以下wrapper之前：
1. SERLObsWrapper (观测包装)
2. ChunkingWrapper (动作分块)

这样确保约束应用在最终的action空间上。

当前timing_belt配置中的顺序：
```python
env = DualAirbotEnv(...)
env = DualAirbotIntervention(...)  # 1. 人类干预
env = DeltaJointActionWrapper(...)  # 2. 相对控制（可选）
env = ActionDerivativeConstraintWrapper(...)  # 3. 导数约束 ← 新增
env = SERLObsWrapper(...)  # 4. 观测包装
env = ChunkingWrapper(...)  # 5. 动作分块
```

## 故障排除

### 如果策略学习缓慢
- 尝试增加 `action_derivative_scale`（例如1.2或1.5）
- 检查演示数据是否足够多样化

### 如果机器人运动不够平滑
- 减小 `action_derivative_scale`（例如0.9或0.8）
- 重新录制更平滑的演示

### 如果提示找不到stats文件
- 确保路径是相对于运行脚本的目录
- 使用绝对路径：`action_derivative_stats_path = os.path.abspath("./experiments/timing_belt/action_derivative_stats.json")`

## 文件说明

- **serl_robot_infra/airbot_env/envs/wrappers.py**: 包含 `ActionDerivativeConstraintWrapper` 类
- **examples/compute_action_derivative_stats.py**: 提取演示数据的导数统计信息
- **examples/experiments/timing_belt/config.py**: 任务配置，包含wrapper的使用示例
