# AirScape Candidate Trajectory Scorer

本项目实现了一个基于 **AirScape / CogVideoX-I2V** 的候选轨迹评分器，用于和无人机规划器模块对接。规划器输入多条候选轨迹后，评分器会为每条轨迹生成未来第一视角视频，并基于目标接近程度、安全性、语义一致性、时序连续性、动力学合法性和不确定性进行综合评分，最终返回最优轨迹编号 `best_traj_id`。

当前版本主要用于 Demo 验证：已经打通从“候选轨迹输入 → motion prompt 构造 → AirScape 视频生成 → 多维评分 → 最优轨迹返回”的完整流程。

---

## 1. 项目目标

本项目面向“基于世界模型的无人机自主行进”任务，目标是将世界模型作为规划器之后的 **候选轨迹评估模块**。

规划器负责生成多条候选轨迹：

```text
traj_0, traj_1, ..., traj_K
```

世界模型评分器负责回答：

```text
哪条候选轨迹在未来更可能安全、稳定、接近目标，并符合任务意图？
```

因此，本项目提供如下核心接口：

```python
score_trajectories(request) -> response
```

输入：

- 当前无人机第一视角图像
- 历史帧
- 当前 UAV 状态
- 任务目标
- 长尾环境条件
- 多条候选轨迹

输出：

- 最优轨迹编号 `best_traj_id`
- 轨迹排序 `ranked_traj_ids`
- 每条轨迹的综合得分与分项得分
- 每条轨迹对应的预测视频路径

---

## 2. 整体流程

```text
规划器输出多条候选轨迹
        ↓
数值轨迹转 AirScape motion prompt
        ↓
AirScape 对每条轨迹生成未来视频
        ↓
取预测视频最后一帧，与 goal image 计算 goal_score
        ↓
计算 safety_score / semantic_score / temporal_score / dynamics_score / uncertainty
        ↓
根据加权公式计算 total_score
        ↓
返回 best_traj_id 与每条轨迹分项评分
```

当前代码入口文件：

```text
/data0/llj/codex-airscape/AirScape.code/tools/score_candidate_trajectories.py
```

---

## 3. 输入格式

评分器输入为一个 JSON request。

示例：

```json
{
  "current_rgb": "path/to/current.jpg",
  "history_rgbs": ["path/to/t-2.jpg", "path/to/t-1.jpg"],
  "uav_state": {
    "x": 0.0,
    "y": 0.0,
    "z": 10.0,
    "yaw": 1.57,
    "vx": 0.0,
    "vy": 0.0,
    "vz": 0.0
  },
  "goal": {
    "type": "image_text",
    "goal_image": "path/to/goal.jpg",
    "instruction": "fly to the red roof"
  },
  "tail_condition": "low_light",
  "candidate_trajectories": [
    {
      "traj_id": 0,
      "waypoints": [
        {"dt": 0.5, "dx": 1.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}
      ]
    }
  ]
}
```

### 字段说明

| 字段 | 含义 |
|---|---|
| `current_rgb` | 当前无人机第一视角图像，用作 AirScape I2V 的条件图像 |
| `history_rgbs` | 历史图像帧，当前代码保留该字段，但暂未实际使用 |
| `uav_state` | 当前无人机状态，当前主要使用 `z` 高度计算安全分数 |
| `goal` | 任务目标，当前使用 `goal_image` 与 `instruction` |
| `tail_condition` | 长尾环境条件，例如 `clear_daytime`、`low_light`、`fog`、`rain` |
| `candidate_trajectories` | 规划器提供的候选轨迹列表 |

### Waypoint 字段说明

| 字段 | 含义 |
|---|---|
| `dt` | 当前动作段持续时间，单位为秒 |
| `dx` | 机体坐标系下前后位移；`dx > 0` 表示前进，`dx < 0` 表示后退 |
| `dy` | 机体坐标系下左右位移；`dy > 0` 表示右移，`dy < 0` 表示左移 |
| `dz` | 高度变化；`dz > 0` 表示上升，`dz < 0` 表示下降 |
| `dyaw` | 偏航角变化；`dyaw > 0` 表示右转，`dyaw < 0` 表示左转 |

---

## 4. Demo 输入

本次 Demo 使用 AirScape 测试样本。

当前图像：

```text
/data0/llj/codex-airscape/test_inputs/00847_urbanvideo_test.jpg
```

目标图像：

```text
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/goal_last_frame.jpg
```

目标图像来源于：

```text
/data0/llj/codex-airscape/datasets/AirScape-Dataset/test/00847_urbanvideo_test.mp4
```

其中，目标图像为该测试视频的最后一帧。

本次 Demo 中规划器提供 2 条候选轨迹。

### 候选轨迹 0

```json
{
  "traj_id": 0,
  "waypoints": [
    {"dt": 0.5, "dx": 1.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0},
    {"dt": 0.5, "dx": 1.0, "dy": 0.2, "dz": 0.0, "dyaw": 0.1}
  ]
}
```

含义：向前飞 2.0 米，轻微右移 0.2 米，并右转 0.1 弧度。

### 候选轨迹 1

```json
{
  "traj_id": 1,
  "waypoints": [
    {"dt": 0.5, "dx": -0.4, "dy": -0.2, "dz": 0.1, "dyaw": -0.1},
    {"dt": 0.5, "dx": -0.4, "dy": -0.2, "dz": 0.1, "dyaw": -0.1}
  ]
}
```

含义：向后飞 0.8 米，左移 0.4 米，上升 0.2 米，并左转 0.2 弧度。

---

## 5. 核心模块

### 5.1 `trajectory_to_motion_prompt`

```python
trajectory_to_motion_prompt(traj, goal, tail_condition=None) -> str
```

作用：将规划器输出的数值轨迹转换为 AirScape 可理解的自然语言 motion prompt。

输入：

- `traj`：单条候选轨迹
- `goal`：目标信息，包含 `instruction`
- `tail_condition`：场景条件，例如 `clear_daytime` 或 `low_light`

输出示例：

```text
The drone will fly forward 2.0 meters, move right 0.2 meters,
turn right 0.1 radians, while following the instruction:
fly toward the urban road and inspect the buildings ahead.
The scene condition is clear_daytime.
```

---

### 5.2 `airscape_generate_video`

```python
airscape_generate_video(
    current_rgb,
    motion_prompt,
    output_path,
    model_path,
    transformer_path,
    steps,
    guidance_scale,
    seed,
    gpu
) -> Path
```

作用：调用 AirScape / CogVideoX-I2V，根据当前图像和 motion prompt 生成未来视频。

主要输入：

- `current_rgb`：当前 RGB 图像
- `motion_prompt`：由轨迹转换得到的自然语言动作描述
- `output_path`：生成视频保存路径
- `steps`：扩散推理步数
- `seed`：随机种子，用于不确定性多采样

输出：

```text
path/to/traj_i_pred.mp4
```

---

### 5.3 `generate_candidate_videos`

```python
generate_candidate_videos(request, output_dir, ...) -> dict
```

作用：对 request 中每条候选轨迹生成一个主预测视频。

输出示例：

```json
{
  "motion_prompts": {
    "0": "The drone will ..."
  },
  "predicted_videos": {
    "0": "path/to/traj_0_pred.mp4"
  }
}
```

说明：

- `predicted_videos` 中只保存最终 response 返回给规划器的主视频。
- 用于不确定性计算的额外采样视频保存在 `uncertainty_samples/` 文件夹中，不放入主 `predicted_videos` 字段。

---

### 5.4 `check_dynamics`

```python
check_dynamics(traj, max_v, max_yaw_rate, max_acc) -> (dynamics_score, rejected, details)
```

作用：检查候选轨迹是否满足基本动力学约束。

检查内容：

- 最大速度 `max_velocity`
- 最大偏航角速度 `max_yaw_rate`
- 最大加速度 `max_acceleration`

如果超过阈值，该轨迹会被标记为 `rejected = true`。

---

### 5.5 `compute_safety_score`

```python
compute_safety_score(traj, uav_state, dynamics_rejected, min_altitude) -> float
```

作用：计算轨迹安全分数。

当前实现：

- 如果动力学非法，则 `safety_score = 0.0`
- 如果轨迹导致无人机高度低于 `min_altitude`，则 `safety_score = 0.2`
- 否则 `safety_score = 0.9`

说明：

当前 Demo 使用轨迹几何规则和最低高度作为安全判断。正式版本可进一步替换为：

- 深度估计
- 碰撞预测头
- AirSim collision label
- 障碍物距离估计

---

### 5.6 `compute_temporal_score`

```python
compute_temporal_score(pred_video) -> float
```

作用：评估生成视频是否连续，是否存在明显跳变。

当前实现：

- 抽取视频帧
- 计算相邻帧 RGB 差异
- 平均差异和波动越大，`temporal_score` 越低

正式版本可替换为 VBench、VideoScore 或其他视频质量评价模块。

---

### 5.7 `compute_semantic_score`

```python
compute_semantic_score(pred_video, instruction, motion_prompt) -> float
```

作用：评估生成视频是否符合任务指令和 motion prompt。

当前实现：

- 尚未接入 VLM
- 使用 `instruction` 和 `motion_prompt` 的文本 token 重叠作为启发式分数

后续建议：

- 接入 VLM
- 判断 predicted video 是否符合 instruction
- 判断 predicted video 是否符合 motion intention

---

### 5.8 `compute_uncertainty`

```python
compute_uncertainty(goal_scores) -> float
```

作用：估计同一条候选轨迹在多次生成下的不确定性。

当前实现：

```text
同一条轨迹生成 N 次视频
每个视频计算 goal_score
uncertainty = variance(goal_scores)
```

本次 Demo 设置：

```text
--uncertainty-samples 3
```

因此，每条轨迹共生成 3 个视频用于不确定性计算：

- 1 个主视频
- 2 个额外采样视频

对于 2 条轨迹，总共生成 6 个视频参与不确定性计算。

---

### 5.9 `score_generated_videos`

```python
score_generated_videos(request, generated, output_dir, ...) -> response
```

作用：对已经生成的视频进行评分。

计算指标包括：

- `goal_score`
- `safety_score`
- `semantic_score`
- `temporal_score`
- `dynamics_score`
- `uncertainty`
- `total_score`
- `rejected`

---

### 5.10 `score_trajectories`

```python
score_trajectories(request) -> response
```

作用：完整评分入口。

内部流程：

1. 对每条候选轨迹生成主预测视频
2. 对每条候选轨迹额外生成 uncertainty sample 视频
3. 计算所有分项得分
4. 根据加权公式计算 `total_score`
5. 返回 `best_traj_id`

---

## 6. 评分公式

当前实现的综合评分公式为：

```text
total_score =
  0.30 * goal_score
+ 0.25 * safety_score
+ 0.15 * semantic_score
+ 0.10 * temporal_score
+ 0.10 * dynamics_score
- 0.10 * uncertainty
```

其中：

| 分项 | 含义 |
|---|---|
| `goal_score` | 预测终点是否接近目标图像 |
| `safety_score` | 轨迹是否满足安全要求 |
| `semantic_score` | 预测结果是否符合任务指令和 motion prompt |
| `temporal_score` | 生成视频是否时序连续、稳定 |
| `dynamics_score` | 轨迹是否满足无人机动力学约束 |
| `uncertainty` | 同一轨迹多次生成结果的不确定性 |

### 拒绝规则

如果满足以下任意条件，则轨迹会被标记为 `rejected = true`：

```text
dynamics 非法
safety_score < 0.3
semantic_score < 0.2
uncertainty > 0.7
```

---

## 7. 输出格式

标准输出格式如下：

```json
{
  "best_traj_id": 0,
  "ranked_traj_ids": [0, 2, 1],
  "scores": [
    {
      "traj_id": 0,
      "total_score": 0.82,
      "goal_score": 0.75,
      "safety_score": 0.90,
      "semantic_score": 0.80,
      "temporal_score": 0.78,
      "dynamics_score": 0.95,
      "uncertainty": 0.12,
      "rejected": false
    }
  ],
  "predicted_videos": {
    "0": "path/to/traj_0_pred.mp4"
  }
}
```

---

## 8. 运行命令

本次 Demo 命令如下：

```bash
cd /data0/llj/codex-airscape/AirScape.code

TORCH_HOME=/data0/llj/codex-airscape/.cache/torch \
/data0/llj/codex-airscape/env/bin/python tools/score_candidate_trajectories.py \
  --request /data0/llj/codex-airscape/outputs/trajectory_scorer_demo/request.json \
  --score \
  --steps 1 \
  --gpu 0 \
  --uncertainty-samples 3 \
  --goal-metric pixel \
  --score-device cpu \
  --output-dir /data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3 \
  --output /data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/result.json
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--steps 1` | Demo 快速验证流程使用，证明完整流程可以跑通 |
| `--steps 50` | 正式推理建议使用更高扩散步数 |
| `--uncertainty-samples 3` | 每条轨迹生成 3 个视频，用于不确定性估计 |
| `--goal-metric pixel` | 当前环境没有 LPIPS / DreamSim，因此使用 pixel fallback |
| `--score-device cpu` | 评分阶段使用 CPU |

正式实验建议在安装 LPIPS / DreamSim 后，将 `goal-metric` 替换为更合理的视觉相似度指标。

---

## 9. Demo 输出结果

输出文件：

```text
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/result.json
```

实际输出：

```json
{
  "best_traj_id": 1,
  "ranked_traj_ids": [1, 0],
  "scores": [
    {
      "traj_id": 0,
      "total_score": 0.8166554298343004,
      "goal_score": 0.9081947654485703,
      "safety_score": 0.9,
      "semantic_score": 1.0,
      "temporal_score": 0.6919741118904414,
      "dynamics_score": 1.0,
      "uncertainty": 0.000004109893147672159,
      "rejected": false
    },
    {
      "traj_id": 1,
      "total_score": 0.8209907370139887,
      "goal_score": 0.904625341296196,
      "safety_score": 0.9,
      "semantic_score": 1.0,
      "temporal_score": 0.7460345666886495,
      "dynamics_score": 1.0,
      "uncertainty": 0.000003220437350323932,
      "rejected": false
    }
  ],
  "predicted_videos": {
    "0": "/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/traj_0_pred.mp4",
    "1": "/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/traj_1_pred.mp4"
  }
}
```

最终选择结果：

```text
best_traj_id = 1
```

---

## 10. 结果解释

两条轨迹均未被拒绝：

```text
traj_0 rejected = false
traj_1 rejected = false
```

两条轨迹的动力学均合法：

```text
traj_0 dynamics_score = 1.0
traj_1 dynamics_score = 1.0
```

候选轨迹 0 的目标相似度略高：

```text
traj_0 goal_score = 0.9082
traj_1 goal_score = 0.9046
```

但候选轨迹 1 的时序连续性更好：

```text
traj_0 temporal_score = 0.6920
traj_1 temporal_score = 0.7460
```

经过完整公式加权后：

```text
traj_0 total_score = 0.8167
traj_1 total_score = 0.8210
```

因此，系统最终选择：

```text
best_traj_id = 1
```

这说明当前评分器不是只根据目标图像相似度选轨迹，而是综合考虑目标接近程度、视频连续性、安全性、动力学合法性和不确定性。

---

## 11. 生成文件

主预测视频：

```text
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/traj_0_pred.mp4
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/traj_1_pred.mp4
```

不确定性采样视频：

```text
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/uncertainty_samples/traj_0_sample_1_pred.mp4
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/uncertainty_samples/traj_0_sample_2_pred.mp4
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/uncertainty_samples/traj_1_sample_1_pred.mp4
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/uncertainty_samples/traj_1_sample_2_pred.mp4
```

最终帧目录：

```text
/data0/llj/codex-airscape/outputs/trajectory_scorer_demo/run_steps1_samples3/final_frames/
```

生成视频格式：

| 属性 | 数值 |
|---|---|
| 分辨率 | 720 × 480 |
| 帧数 | 49 |
| FPS | 8 |
| 时长 | 6.125 秒 |

---

## 12. 耗时与工程优化

本次 Demo 设置：

```text
候选轨迹数 K = 2
uncertainty_samples N = 3
扩散步数 steps = 1
总生成视频数 = K * N = 6
```

当前实现中，每生成一个视频都会重新加载一次 AirScape / CogVideoX 模型。因此耗时主要来自两部分：

```text
模型加载时间 + diffusion 推理时间
```

本次观察到：

- `steps = 1` 时，单次 diffusion 推理约 20–23 秒。
- 第一次模型加载可能需要数分钟。
- 后续由于系统缓存，加载会变快，但仍存在重复加载开销。

正式推理若使用：

```text
K 条轨迹
N = 3 次 uncertainty sample
steps = 50
```

则需要生成：

```text
K * N 个视频
```

在当前 sequential offload 设置下，单个 `steps = 50` 视频可能达到 8–12 分钟级别。如果 `K = 3, N = 3`，总共需要生成 9 个视频，整体耗时可能达到 1–2 小时甚至更长。

这说明当前瓶颈不是评分公式，而是 AirScape 大模型的视频生成过程。

### 优化方向

后续可从以下方向优化：

1. **模型只加载一次**  
   在同一个 Python 进程中循环生成所有候选轨迹视频，避免重复加载模型。

2. **并行生成候选轨迹视频**  
   多 GPU 环境下，不同候选轨迹可以并行推理。

3. **减少 sequential offload**  
   在显存允许的情况下，降低 offload 带来的速度损失。

4. **两级筛选**  
   先用规则或轻量模型筛选 Top-M，再对 Top-M 调用 AirScape 生成视频。

5. **蒸馏轻量 scorer**  
   将 AirScape 视频评分结果作为 teacher signal，训练轻量轨迹评分器，实现实时推理。

---

## 13. 当前局限性

当前 Demo 已经跑通完整流程，但部分模块仍是初版启发式实现。

| 模块 | 当前实现 | 后续替换方向 |
|---|---|---|
| `goal_score` | pixel fallback | LPIPS / DreamSim / DINO feature similarity |
| `safety_score` | 最低高度 + 动力学规则 | 深度估计 / collision head / AirSim 碰撞标签 |
| `semantic_score` | 文本 token 重叠 | VLM 视频语义判断 |
| `temporal_score` | 相邻帧 RGB 差异 | VBench / VideoScore / 时空判别器 |
| `uncertainty` | 多次生成的 `goal_score` 方差 | ensemble variance / latent variance / risk uncertainty |

---

## 14. 与规划器的对接方式

规划器组只需要提供标准 request，评分器返回 response。

规划器侧输入：

```text
current_rgb
uav_state
goal
candidate_trajectories
```

评分器侧输出：

```text
best_traj_id
ranked_traj_ids
scores
predicted_videos
```

建议规划器优先统一以下内容：

1. 候选轨迹格式：`dx, dy, dz, dyaw, dt`
2. 坐标系：机体系还是世界系
3. 目标形式：目标图像、目标坐标、语言指令，或三者组合
4. 是否允许返回“无安全轨迹”
5. 是否能从 AirSim 获取碰撞、深度、目标距离变化和真实执行结果

---

## 15. 一句话总结

本 Demo 实现了一个基于 AirScape 的候选轨迹评分器：给定当前无人机图像和多条规划器候选轨迹，系统会将数值轨迹转换为 AirScape motion prompt，生成每条轨迹的未来第一视角视频，并基于目标相似度、安全性、语义一致性、时序连续性、动力学合法性和不确定性计算综合分数，最终返回最优轨迹编号。
