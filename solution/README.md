# 求解器说明

`solution/` 存放调度策略实现。每个求解器读取同一个环境状态和合法动作
掩码，并决定下一步执行哪类作业、操作哪架飞机。

求解器的统一调用入口位于 `scripts/solve.py`，环境状态转移位于
`env/carrier_aircraft_env.py`。因此：

- `env/` 定义问题及约束；
- `solution/` 决定下一步动作；
- `scripts/` 负责创建环境、循环调用求解器和汇总结果。

## 统一接口

所有求解器均实现：

```python
choose_action() -> Optional[Dict[str, int]]
```

有合法动作时返回：

```python
{"high_level": action_id, "aircraft_id": aircraft_id}
```

高层动作编码如下：

| ID | 动作 | 含义 |
|---:|---|---|
| 0 | `R` | 开始回收 |
| 1 | `F` | 开始加油 |
| 2 | `M` | 开始挂弹 |
| 3 | `L` | 开始放飞 |

没有合法动作时返回 `None`，由环境将时间推进到下一个事件。求解器不得
绕过 `env.get_action_mask()` 生成非法动作。

## 已实现求解器

| CLI 名称 | 类 | 类型 | 是否训练 | 主要用途 |
|---|---|---|---|---|
| `random` | `RandomSolver` | 随机策略 | 否 | 最弱基线、环境正确性检查 |
| `fifo` | `FIFOSolver` | 经典派工规则 | 否 | 最长等待优先基线 |
| `spt` | `SPTSolver` | 经典派工规则 | 否 | 最短作业时间优先基线 |
| `edd` | `EDDSolver` | 经典派工规则 | 否 | 最早波次截止时间优先基线 |
| `heuristic` | `WaveHeuristicSolver` | 规则启发式 | 否 | 当前主要非学习基线 |
| `sampled` | `SampledRandomSolver` | 随机采样规划 | 否 | 检验多次随机搜索能带来的提升 |
| `cp_sat` | `CPSATSolver` | 滚动整数优化 | 否 | 优化当前批次的资源分配 |
| `rl` | `RLSolver` | 神经网络策略 | 是 | 加载 PPO checkpoint 进行推理 |

## RandomSolver

文件：`solution/random_solver.py`

决策过程：

1. 从当前合法的高层动作中均匀随机选择一种作业；
2. 从该作业对应的合法飞机中均匀随机选择一架。

这里不是对全部合法 `(动作, 飞机)` 二元组均匀采样。候选飞机数量较少的
动作类型与候选飞机数量较多的动作类型，获得相同的高层选择概率。

特点：

- 不使用波次截止时间、剩余工时或资源紧张程度；
- 固定环境种子和求解器种子时可以复现；
- 适合验证环境能否跑完整个仿真，也是其他方法的最低性能基线。

运行：

```bash
python -m scripts.solve --solver random --seed 7 --runs 10
```

## FIFO、SPT 与 EDD

文件：`solution/priority_rule_solver.py`

三个经典派工规则共用 `PriorityRuleSolver`：

- `FIFO`：等待时间最长的合法作业优先；
- `SPT`：预计处理时间最短的合法作业优先；
- `EDD`：所属飞机下一放飞窗口的截止时间最早者优先。

它们不进行搜索或训练，计算量与当前合法动作数量近似线性相关。三者为
启发式和学习方法提供标准、可解释的调度基线。

```bash
python -m scripts.solve --solver fifo --runs 10
python -m scripts.solve --solver spt --runs 10
python -m scripts.solve --solver edd --runs 10
```

## WaveHeuristicSolver

文件：`solution/heuristic_solver.py`

这是当前主要的规则求解器。高层优先级为：

```text
可放飞时立即放飞
    > 加油或挂弹
    > 回收
    > 无动作时推进时间
```

加油与挂弹候选飞机通过剩余工作量和波次松弛时间排序：

```text
remaining = max(fuel_work, arm_work)
slack = time_to_launch_deadline - remaining
```

主要规则：

- 回收与放飞同时可执行时，先启动回收，再使用独立起飞位并行放飞；
- 优先处理仍可能赶上目标波次的飞机；
- deadline 更近、剩余工作量更小的飞机优先；
- 挂弹排序考虑弹药数量和停机位转运时间；
- 人员不足时，在加油与挂弹之间保持一定并发平衡；
- 已满足放飞条件的飞机立即进入放飞队列。

优点是速度快、行为可解释，不需要训练。局限是规则固定，只利用局部
slack 估计，无法系统搜索跨波次的长期资源分配。

运行：

```bash
python -m scripts.solve --solver heuristic --runs 10
```

## SampledRandomSolver

文件：`solution/sampled_random_solver.py`

该求解器在第一次决策时：

1. 深拷贝当前环境；
2. 在每个副本中运行一条完整的随机动作序列；
3. 共采样 `samples` 条轨迹；
4. 选择得分最高的完整轨迹；
5. 在真实环境中依次重放该轨迹。

当前评分顺序是：

```text
先增加 completed sorties
再减少 missed sorties
最后减少仿真完成时间
```

即源码中的 `(completed, -missed, -time)`，与项目最大化完成放飞架次的
主目标一致。

它不是逐步滚动规划，而是在起点一次性选定完整轨迹。采样数越大，计算
量和内存消耗越高，而且效果仍受随机轨迹覆盖范围限制。

运行：

```bash
python -m scripts.solve --solver sampled --sampled-samples 30
```

## CPSATSolver

文件：`solution/cp_sat_solver.py`

这是一个基于 OR-Tools 的滚动批次优化基线：

1. 放飞和回收使用独立资源，因此有合法动作时立即派发；
2. 对当前可执行的加油与挂弹作业建立 0-1 整数模型；
3. 约束当前加油服务器、弹药转运车、下层升降机和人员容量；
4. 优先让更多飞机同时获得完整的加油与挂弹资源；
5. 在同等条件下优先处理截止时间更近、预计工时更短的飞机；
6. 执行一个动作后，根据新状态重新求解。

它优化的是当前决策时刻的资源分配，不是整个随机仿真周期的全局最优
解，因此应称为滚动 CP-SAT 基线。

```bash
python -m scripts.solve --solver cp_sat --cp-sat-max-time 0.05
```

## RLSolver

文件：`solution/rl_solver.py`

`RLSolver` 是训练后策略的推理包装器。它本身不训练模型，训练过程位于
`scripts/train_rl.py`。

推理过程：

1. 将每架飞机的 18 维特征和 18 维全局特征编码为张量；
2. 网络输出 4 个高层动作 logits、每架飞机的低层 logits 和状态价值；
3. 使用 action mask 屏蔽非法动作；
4. 先选择作业类型，再选择飞机；
5. 确定性模式取最大 logit，随机模式按策略分布采样。

模型结构位于 `rl/model.py`，采用共享飞机编码器、mean/max pooling、
高层动作头、飞机选择头和价值头。

运行：

```bash
python -m scripts.solve \
    --solver rl \
    --checkpoint checkpoints/rl_policy.pt
```

注意：当前实现不会强制 checkpoint 存在。路径为空或文件不存在时，会
使用随机初始化网络进行推理，这不代表训练后的 RL 效果。正式评估必须
提供有效 checkpoint，并记录训练配置和随机种子。

## 如何选择

| 场景 | 推荐求解器 |
|---|---|
| 检查环境是否可运行 | `random` |
| 对比经典派工规则 | `fifo`、`spt`、`edd` |
| 获得快速、较强的规则基线 | `heuristic` |
| 测试随机搜索上限 | `sampled` |
| 测试滚动整数资源分配 | `cp_sat` |
| 评估训练后的神经网络策略 | `rl` |

所有方法应在完全相同的环境参数和随机种子上比较，主指标统一使用
`total_sorties_completed`。`total_missed_sorties`、运行步数和计算耗时
作为辅助分析指标。

统一基准命令：

```bash
python -m scripts.benchmark_non_rl
```

默认使用 60、80、100、120 分钟波次间隔，每种配置运行 12 个完整波次和
10 个相同随机种子，结果写入 `outputs/non_rl_benchmark.csv`。
