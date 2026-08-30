# 舰载机多波次调度

基于离散事件仿真的舰载机回收、整备与放飞调度项目，包含随机策略、
启发式策略、采样规划和强化学习求解器。

---

## 1. 问题概述

航母上有两组飞机（A 组和 B 组），交替波次出动。完整任务周期：

```
放飞 → 升空执行任务 → 回收 → 加油 ‖ 挂弹 → 再次放飞
```

**优化目标**：在固定的 `simulation_duration`（默认 720 时间单位）内，
最大化完成的放飞架次。错失架次（missed sorties）作为评估指标。

---

## 2. 波次机制

| 波次 | 放飞组 | 回收组 |
|------|--------|--------|
| 0    | A      | 无     |
| 1    | B      | A      |
| 2    | A      | B      |
| 3    | B      | A      |
| ...  | 交替   | 交替   |

- 波次间隔：`wave_interval`（默认 120.0）
- 第 0 波 A 组直接起飞（初始已完成整备，处于待放飞状态）
- 新波次开始时，上一波应放飞但未起飞的飞机记为 **错失架次**

---

## 3. 飞机状态机

每架飞机有四个状态维度：

| 维度             | 取值     | 含义                        |
|------------------|----------|-----------------------------|
| `recovery_status` | 0/1/2   | 等待回收 / 回收中 / 已完成  |
| `fuel_status`     | 0/1/2   | 等待加油 / 加油中 / 已完成  |
| `arm_status`      | 0/1/2   | 等待挂弹 / 挂弹中 / 已完成  |
| `launch_status`   | 0/1/2/3 | 未就绪 / 待放飞 / 放飞中 / 已完成 |

**关键规则：**
- 加油和挂弹**可以并行**（同一架飞机可同时处于加油中和挂弹中）
- 放飞**必须**等加油和挂弹均完成（`fuel_status=2` 且 `arm_status=2`）
- 放飞完成后，飞机变为升空状态，待其所在组返回时进入回收队列

---

## 4. 资源配置

| 资源         | 数量 | 说明                       |
|--------------|------|----------------------------|
| 回收通道     | 1    | 独占，一次只能回收一架     |
| 放飞通道     | 1    | 独占，一次只能放飞一架     |
| 加油服务器   | 20   | 可并行                     |
| 挂弹车       | 10   | 可并行                     |
| 保障人员     | 50   | 加油耗 4 人/架，挂弹耗 4 人/架 |

---

## 5. 动作空间（分层架构）

### 高层动作

| ID | 名称 | 含义 |
|----|------|------|
| 0  | R    | 回收 |
| 1  | F    | 加油 |
| 2  | M    | 挂弹 |
| 3  | L    | 放飞 |

### 低层动作

选好高层动作后，从候选集合中选择一架具体飞机。环境通过
`get_action_mask()` 提供完整的 mask。

候选集合：
- **R**：当前回收组中 `pending_recovery=true` 的飞机
- **F**：已完成回收、未加油的飞机
- **M**：已完成回收、未挂弹的飞机
- **L**：当前放飞组中加油和挂弹均完成、待放飞的飞机

---

## 6. 事件驱动机制

环境采用**事件驱动**而非固定时间步长。当无可行动作时，自动推进到下一事件。

| 事件           | 触发条件                                   |
|----------------|--------------------------------------------|
| `wave_start`   | 每隔 `wave_interval` 时间单位              |
| `recover_done` | 回收开始后 1.0 时间单位                    |
| `fuel_done`    | 加油开始后 N(20, 3) 时间单位               |
| `arm_done`     | 挂弹开始后 数量 × N(5, √2) 时间单位（累加）|
| `launch_done`  | 放飞开始后 1.0 时间单位                    |

事件通过 `heapq` 优先队列管理。

---

## 7. 时效参数

| 参数                     | 默认值 | 说明                 |
|--------------------------|--------|----------------------|
| `recovery_time`          | 1.0    | 回收耗时             |
| `launch_time`            | 1.0    | 放飞耗时             |
| `fuel_time_mean`         | 20.0   | 加油时间均值         |
| `fuel_time_std`          | 3.0    | 加油时间标准差       |
| `arm_unit_time_mean`     | 5.0    | 单枚弹药挂载时间均值 |
| `arm_unit_time_variance` | 2.0    | 单枚弹药挂载时间方差 |
| `wave_interval`          | 120.0  | 波次间隔             |
| `simulation_duration`    | 720.0  | 总模拟时长           |

挂弹总时间 = 每枚弹药独立采样的累加。弹药数量在回收完成时从
[1, 2, 3, 4] 中按概率 [0.3, 0.4, 0.2, 0.1] 随机抽取。

---

## 8. 奖励函数

```
Reward = -α·Δt               （时间惩罚，α=1.0）
        + β_rec·N_rec        （回收奖励，β=0.1）
        + β_fuel·N_fuel      （加油奖励，β=0.2）
        + β_arm·N_arm        （挂弹奖励，β=0.2）
        + β_launch·N_launch  （放飞奖励，β=1.0）
        + terminal_reward    （模拟结束奖励 100.0）
```

- 非法动作惩罚：-5.0
- 空闲惩罚（有可行动作但传入 `None`）：-1.0

---

## 9. 典型执行流程

1. `reset()` → 初始化 A/B 组飞机，启动波次 0（A 组起飞）
2. 智能体选择 `L` + aircraft_id → 放飞 A 组飞机
3. 无可行动作 → 自动推进到下一事件
4. 智能体选择 `R` → `F` / `M`（并行）→ `L` → 等待下一波次
5. 循环直到 `time >= simulation_duration`
6. 输出评估：`total_sorties_completed`、`total_missed_sorties`

---

## 文件结构

```text
.
├── env/          # 事件驱动仿真环境、状态机与配置
├── solution/     # Random、Heuristic、Sampled、RL 求解器
├── rl/           # 观测编码、策略网络、PPO 与 checkpoint
├── scripts/      # 求解、训练、评估和随机基线入口
├── outputs/      # 已有实验 CSV 与图表
├── doc/          # 前置约束、建模说明和项目文档
├── README.md
└── requirements.txt
```

`env/` 只负责定义问题和推进状态；`solution/` 负责根据环境状态选择动作；
`rl/` 只包含强化学习组件；可执行入口统一放在 `scripts/`。

## 运行

从项目根目录执行：

```bash
# 默认启发式求解
python -m scripts.solve

# 其他求解器
python -m scripts.solve --solver random --runs 5
python -m scripts.solve --solver sampled --sampled-samples 30
python -m scripts.solve --solver rl --checkpoint checkpoints/rl_policy.pt

# PPO 训练与评估
python -m scripts.train_rl --checkpoint checkpoints/rl_policy.pt
python -m scripts.evaluate_rl --checkpoint checkpoints/rl_policy.pt --runs 10

# 旧版随机策略明细输出
python -m scripts.random_policy_test --runs 5
```

实验结果建议写入 `outputs/`：

```bash
python -m scripts.solve \
    --solver heuristic \
    --runs 10 \
    --runs-csv outputs/runs.csv \
    --timing-csv outputs/timing.csv \
    --missed-csv outputs/missed.csv
```

## 动作格式

字典格式：

```python
{"high_level": 1, "aircraft_id": 3}
```

元组格式：

```python
(1, 3)
```

当无可用的高层动作时，调用 `step(None)` 推进时间到下一事件。
