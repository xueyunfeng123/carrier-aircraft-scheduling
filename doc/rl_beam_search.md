# RL 引导的短视域 Beam Search

## 目标与边界

本原型在不改动事件驱动环境状态转移的前提下，为已有 checkpoint 增加
推理期搜索。搜索只负责从当前合法动作中选出下一步动作，真实执行仍通过
`CarrierAircraftSchedulingEnv.step()` 完成。

当前版本刻意不使用 checkpoint 的 value head。叶节点按以下词典序评分：

1. 环境累计完成架次；
2. 既有 `readiness_potential`，即已完成架次和已完成保障工序；
3. 负的累计缺额；
4. checkpoint 给出的动作序列对数概率，仅用于最终平局裁决。

这使第一版实验可以检验“策略缩枝 + 环境真实模拟”本身是否有效，不把
收益归因于尚未校准的 value。

## 与原始文献的关系

1. Silver et al., *Mastering the game of Go with deep neural networks and
   tree search*, Nature 529, 484-489 (2016),
   [doi:10.1038/nature16961](https://doi.org/10.1038/nature16961)。
   AlphaGo 用 policy 缩小搜索宽度，并用 value/rollout 评价局面。本原型
   采用 policy 缩枝，但以调度环境 rollout 和 readiness 替代 value。
2. Anthony, Tian, and Barber, *Thinking Fast and Slow with Deep Learning and
   Tree Search* (2017), [arXiv:1705.08439](https://arxiv.org/abs/1705.08439)。
   ExIt 将慢速搜索专家与快速神经网络学徒交替改进。当前只实现推理期
   “学徒引导专家搜索”，尚未把搜索动作蒸馏回策略。
3. Hamrick et al., *Combining Q-Learning and Search with Amortized Value
   Estimates*, ICLR 2020,
   [arXiv:1912.02807](https://arxiv.org/abs/1912.02807)。
   SAVE 用学习到的 Q prior 引导小预算 MCTS，并将搜索估值摊销回 Q 网络。
   本原型同样关注小搜索预算，但没有 Q prior、MCTS backup 或搜索后训练。
4. Choo et al., *Simulation-guided Beam Search for Neural Combinatorial
   Optimization*, NeurIPS 2022,
   [arXiv:2207.06190](https://arxiv.org/abs/2207.06190)。
   SGBS 用神经策略提出候选、用 simulation rollout 选择固定宽度 beam。
   这是本实现最直接的方法来源；差异在于本问题是随机事件驱动调度，
   搜索每个真实决策后重新规划。

因此代码名称使用 `RLBeamSearchSolver`，不声称完整复现 AlphaGo、ExIt、
SAVE 或论文版 SGBS。

## 搜索过程

`RLSolver.rank_actions(K, env)` 按三层自回归分布
`P(作业) P(飞机|作业) P(目标|作业,飞机)` 进行分层 Top-K 展开，再按联合
概率排序。每一层都先应用环境已有 mask；返回动作再次交给环境合法性检查。

`RLBeamSearchSolver` 在每次真实决策时：

1. 深拷贝当前环境，并替换副本 RNG；
2. 对每个 beam 节点扩展策略 Top-K 合法动作；
3. 在候选副本中调用真实 `env.step()`；
4. 用贪心 checkpoint 策略向前模拟固定事件数；
5. 按完成架次/readiness/缺额裁剪到固定 beam width；
6. 只把最佳序列的第一个动作返回真实环境，然后重新规划。

默认参数是小预算原型：`beam_width=2`、`expansion_k=2`、
`search_depth=2`、`rollout_events=2`、`rollout_samples=1`。

## 无未来信息泄漏约束

搜索遵循以下硬约束：

- 不读取、排序或筛选未来扰动日程；
- 不读取真实环境 RNG 状态来产生未来样本；
- 规划副本在执行第一个候选动作前使用
  `search_seed + decision_index` 派生的独立 RNG；
- rollout 只调用公开的策略编码和正常 `env.step()`，不直接弹出或查看
  `event_queue`；
- 已经开始的作业剩余时间属于当前 observation，允许环境按既定完成事件
  推进；
- 规划不修改真实环境状态或 RNG，真实环境只执行最终选中的一个动作。

当前基础环境没有动态故障日程。后续若重新引入扰动，必须让规划副本从
训练分布独立采样，不能复制 episode 已预生成但尚未观测的扰动列表。

## 命令

单次运行：

```bash
python -m scripts.solve \
  --solver rl_beam \
  --checkpoint checkpoints/rl_learned_prior_ppo.pt \
  --rl-device cuda \
  --beam-width 2 \
  --beam-expansion-k 2 \
  --beam-depth 2 \
  --beam-rollout-events 2 \
  --beam-rollout-samples 1 \
  --seed 41001
```

配对基准：

```bash
python -m scripts.benchmark_non_rl \
  --solvers heuristic cp_sat \
  --rl-checkpoint checkpoints/rl_learned_prior_ppo.pt \
  --rl-beam-checkpoint checkpoints/rl_learned_prior_ppo.pt \
  --rl-device cuda \
  --seed 41001 --runs 5 --intervals 60 --waves 12 \
  --beam-width 2 --beam-expansion-k 2 \
  --beam-depth 1 --beam-rollout-events 1 \
  --beam-rollout-samples 1 \
  --output outputs/rl_beam_seed41001_5runs.csv \
  --detail-output outputs/rl_beam_seed41001_5runs_detail.csv
```

冻结筛选集只允许 `41001-41005`；不得用 `70001-70050` 做本方向的参数
筛选。

## 5-seed 筛选结果

2026-09-19 在远端 RTX A6000 主机上执行上述小预算配置。汇总结果保存在
`outputs/rl_beam_seed41001_5runs.csv`，逐种子结果保存在
`outputs/rl_beam_seed41001_5runs_detail.csv`。

| 方法 | 平均完成架次 | 标准差 | 平均运行时间 |
|---|---:|---:|---:|
| Heuristic | 118.20 | 0.84 | 59.29 s |
| CP-SAT | 119.40 | 0.55 | 59.60 s |
| RL | 120.00 | 0.71 | 63.03 s |
| RL Beam | 120.00 | 0.71 | 246.01 s |

RL Beam 相对 RL 的逐种子差值为 `0, +1, -1, -1, +1`，即 2 胜 1 平
2 负，均值差为 0，双侧精确符号检验 `p=1.0`。相对 CP-SAT 为
3 胜 2 平、均值 `+0.60`、`p=0.25`；相对 Heuristic 为 5 胜、
均值 `+1.80`、`p=0.0625`。RL Beam 的平均运行时间是 RL 的
3.90 倍。

CSV 校验覆盖 4 个求解器、5 个种子和每个 episode 的 12 个波次；
所有记录均满足波次和等于总完成架次，且完成架次与缺额之和为 240。
本次结果只能说明该小预算配置与 RL 持平，不能证明 Beam Search
带来收益，也未达到项目要求的至少 30 个冻结测试种子和 `p < 0.05`
验收条件。
