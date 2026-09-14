# 舰载机多波次调度

基于离散事件仿真的舰载机回收、整备与放飞调度项目，包含随机策略、
启发式策略、采样规划和强化学习求解器。

---

## 1. 问题概述

甲板上有 45 架共享飞机，其中 5 架初始标记为备用。A/B 是交替波次
任务标签，每个波次从当前已完成保障的飞机中动态选择最多 20 架。完整
任务周期：

```
放飞 → 升空执行任务 → 回收 → 加油 ‖ 挂弹 → 再次放飞
```

**优化目标**：在固定的 `simulation_duration`（默认 720 时间单位）内，
最大化完成的放飞架次。错失架次（missed sorties）作为评估指标。

---

## 2. 波次机制

| 波次 | 放飞任务 | 回收任务 |
|------|--------|--------|
| 0    | A 标签，动态选 20 架 | 无 |
| 1    | B 标签，动态选 20 架 | 第 0 波实际放飞飞机 |
| 2    | A 标签，动态选 20 架 | 第 1 波实际放飞飞机 |
| ...  | A/B 标签交替 | 上一波实际放飞飞机 |

- 波次间隔：`wave_interval`（默认 120.0）
- 第 0 波从 45 架已完成整备的飞机中动态选择 20 架；
- 每波启动 20 个放飞席位后停止继续放飞；
- 新波次开始时，未填满的任务席位记为 **错失架次**；
- 初始备用机与其他飞机遵守同一就绪条件，可以直接填充任务席位。

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
- 放飞完成后，飞机变为升空状态，并在下一波进入回收队列

---

## 4. 资源配置

| 资源         | 数量 | 说明                       |
|--------------|------|----------------------------|
| 回收通道     | 1    | 独占，一次只能回收一架     |
| 起飞位       | 4    | 图上的位置容量             |
| 放飞通道     | 1    | 作业容量，一次只能放飞一架 |
| 甲板停机位   | 45   | 起飞区 10、着舰区 10、保障区 25 |
| 通道节点     | 19   | 结构复现实验假设           |
| 加油服务器   | 20   | 可并行                     |
| 挂弹车       | 10   | 可并行                     |
| 保障人员     | 50   | 加油耗 4 人/架，挂弹耗 4 人/架 |

空间图默认启用。飞机放飞前必须取得“停机位到起飞位”的可行路径，
回收时必须取得“着舰跑道到空闲停机位”的可行路径。当前采用整条路径
原子预留，移动完成后释放中间节点；这是缺少真实甲板邻接矩阵时的保守
结构假设，不代表真实甲板几何或逐节点滑行。

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
- **R**：上一波实际放飞且 `pending_recovery=true` 的飞机
- **F**：已完成回收、未加油的飞机
- **M**：已完成回收、未挂弹的飞机
- **L**：共享机队中加油和挂弹均完成的飞机，直到本波填满 20 个席位

---

## 6. 事件驱动机制

环境采用**事件驱动**而非固定时间步长。当无可行动作时，自动推进到下一事件。

| 事件           | 触发条件                                   |
|----------------|--------------------------------------------|
| `wave_start`   | 每隔 `wave_interval` 时间单位              |
| `recover_done` | 回收与跑道至停机位转运完成                 |
| `fuel_done`    | 加油开始后 N(20, 3) 时间单位               |
| `arm_done`     | 挂弹开始后 数量 × N(5, √2) 时间单位（累加）|
| `taxi_to_launch_done` | 到达起飞位并释放中间路径           |
| `launch_done`  | 到达起飞位后 1.0 时间单位                  |

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
| `spatial_graph_enabled`  | true   | 是否启用甲板路径约束 |
| `num_launch_positions`   | 4      | 空间图起飞位节点数   |
| `deck_edge_travel_time`  | 1.0    | 每条结构图边的移动分钟数 |

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

1. `reset()` → 初始化 45 架共享飞机，其中 5 架为初始备用角色
2. 智能体选择 `L` + aircraft_id → 预留滑行路径并动态填充任务席位
3. 无可行动作 → 自动推进到下一事件
4. 智能体选择 `R` → `F` / `M`（并行）→ `L` → 等待下一波次
5. 循环直到 `time >= simulation_duration`
6. 输出评估：`total_sorties_completed`、`total_missed_sorties`

`get_wave_records()` 返回每个波次的任务标签、目标架数、实际启动/完成
飞机 ID、回收飞机 ID、完成架次和未填满席位。

---

## 文件结构

```text
.
├── env/          # 事件驱动仿真环境、状态机与配置
├── solution/     # Random、优先规则、Heuristic、CP-SAT、RL 求解器
├── rl/           # 观测编码、策略网络、PPO 与 checkpoint
├── scripts/      # 求解、训练、评估和随机基线入口
├── outputs/      # 已有实验 CSV 与图表
├── doc/          # 前置约束、建模说明和项目文档
├── README.md
└── requirements.txt
```

`env/` 只负责定义问题和推进状态；`solution/` 负责根据环境状态选择动作；
`rl/` 只包含强化学习组件；可执行入口统一放在 `scripts/`。
各求解器的决策逻辑和适用场景见
[solution/README.md](solution/README.md)。

## 环境配置

项目使用 Python 3.12，推荐使用 Conda：

```bash
conda create \
    --name carrier-aircraft-scheduling \
    --override-channels \
    --channel conda-forge \
    python=3.12 pip -y
conda activate carrier-aircraft-scheduling
python -m pip install -r requirements.txt
```

也可以通过 `environment.yml` 创建：

```bash
conda env create --file environment.yml
conda activate carrier-aircraft-scheduling
```

## 场景配置

环境场景定义位于 `env/scenario.py`，与求解器实现分离。

- `project_core`：45 架共享机队、每波动态选择 20 架的资源调度环境；
- `paper_yoon_2023`：Yoon et al. (2023) 航母出动生成 DES 的结构复现，
  由独立的 `YoonSortieGenerationEnv` 执行。

论文未公开完整甲板邻接矩阵、边距离、移动速度和 FlyPro 明细，因此
`paper_yoon_2023` 明确标记为
`structural_replication_with_explicit_assumptions`，不能称为精确数值复现。
现有调度求解器仍只连接 `project_core`；论文基线先使用其预定运行策略。

```bash
# 查看当前项目场景
python -m scripts.inspect_scenario --profile project_core

# 查看 Yoon 2023 的指定案例
python -m scripts.inspect_scenario \
    --profile paper_yoon_2023 \
    --case case_4_1

# 运行 8 个论文案例的结构复现
python -m scripts.run_yoon_baseline \
    --runs 100 \
    --workers 8 \
    --output outputs/yoon_2023_structural_replication.csv
```

详细复现边界见
[`doc/yoon_2023_environment_spec.md`](doc/yoon_2023_environment_spec.md)。

## 运行

激活环境后，从项目根目录执行：

正式基线统一使用 60 分钟波次、12 个完整波次和单一环境种子
`10007`。这些默认值集中定义在 `scripts/evaluation_defaults.py`；PPO
训练使用独立种子 `7`。

```bash
# 默认启发式求解
python -m scripts.solve

# 其他求解器
python -m scripts.solve --solver random
python -m scripts.solve --solver fifo
python -m scripts.solve --solver spt
python -m scripts.solve --solver edd
python -m scripts.solve --solver cp_sat --cp-sat-max-time 0.05
python -m scripts.solve --solver rl --checkpoint checkpoints/rl_policy.pt

# 关闭空间图，运行匹配对照
python -m scripts.solve --solver heuristic --disable-spatial-graph

# Heuristic 行为克隆预热、PPO 微调与评估
python -m scripts.train_rl --checkpoint checkpoints/rl_policy.pt
python -m scripts.evaluate_rl --checkpoint checkpoints/rl_policy.pt

# 复现三层目标选择 BC+PPO 实验
./scripts/run_rl_target_experiment.sh

# 复现训练/验证/测试隔离的学习式先验实验
./scripts/run_rl_generalization_experiment.sh

# 复现跨负载训练与未见负载测试
./scripts/run_rl_multiload_experiment.sh

# 旧版随机策略明细输出
python -m scripts.random_policy_test

# 全部求解器统一基准
python -m scripts.benchmark_non_rl \
    --rl-checkpoint checkpoints/rl_policy.pt \
    --rl-label rl_bc_ppo \
    --output outputs/all_solver_benchmark_60min_seed10007.csv

# 空间图关闭对照
python -m scripts.benchmark_non_rl \
    --disable-spatial-graph \
    --output outputs/spatial_deck_graph_disabled_control_60min_seed10007.csv
```

RL 策略采用“作业类型→飞机→目标停机位/跑道/车辆”的三层 masked
动作。默认训练从 5 个训练 seed 收集 Heuristic 示范并进行行为克隆，再
使用 PPO 微调。正式实验可通过 `--train-seeds` 和 `--validation-seeds`
显式隔离训练与验证场景；训练器按验证集平均值和最差值保存最佳
checkpoint。纯 PPO 对照可通过 `--bc-episodes 0` 运行。

飞机 SPT 排序和目标最早可达排序的强度可通过 `--low-rank-prior` 与
`--target-rank-prior` 消融。`--adaptive-low-rank-prior` 将固定飞机排序
权重替换为按状态和作业类型学习的门控，详细结果见
`doc/rl_generalization_iteration.md`。

冻结测试 `seed=40001-40020` 上，学习门控 BC+PPO、固定先验 RL 和
CP-SAT 的平均完成架次分别为 120.20、119.75 和 119.50。学习门控策略
相对 CP-SAT 为 14 胜、5 平、1 负。`elite_cp` 教师均值为 119.80；
学习策略均值更高，但该差异尚未达到显著水平。
正式 Heuristic 均值为 117.60，学习门控策略在 20 个种子上全部胜出。
在 47.5-67.5 分钟的 5 个未见负载、共 25 个场景上，跨负载策略相对
Heuristic 平均增加 2.92 架次，为 24 胜、1 平、0 负；相对 CP-SAT
平均增加 0.24 架次，差异不显著。

共享机队将飞机特征从永久 A/B 标记改为初始备用角色，并将全局特征改为
波次填充率和待回收压力；空间图进一步加入目标/车辆集合、候选排序先验
和空闲通道比例，当前观测版本为 9。旧环境训练的 checkpoint 会被明确
拒绝，必须重新执行 BC+PPO 训练。

实验结果建议写入 `outputs/`：

```bash
python -m scripts.solve \
    --solver heuristic \
    --runs 10 \
    --runs-csv outputs/runs.csv \
    --timing-csv outputs/timing.csv \
    --missed-csv outputs/missed.csv \
    --event-log-csv outputs/events.csv
```

环境额外报告：

- `sortie_generation_rate_per_hour`：固定时域内每小时完成的放飞架次；
- `sortie_completion_rate`：已完成架次占当前已开始波次机会数的比例；
- `get_event_log()`：波次、作业开始/完成和 missed sortie 的事件记录。

## 动作格式

字典格式：

```python
{
    "high_level": 1,
    "aircraft_id": 3,
    "vehicle_id": "fuel_2"
}
```

元组格式：

```python
(1, 3)
```

当无可用的高层动作时，调用 `step(None)` 推进时间到下一事件。
