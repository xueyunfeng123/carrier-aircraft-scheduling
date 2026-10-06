# 投稿级配对评估协议

## 1. 目标与原则

主指标仅为固定时域完成放飞架次。所有 solver 必须使用相同 seed、环境
配置、显式扰动日程和 scenario tape。波次不能作为独立统计样本；同一
seed 下的完整 profile/load 矩阵先聚合，再进行配对检验。

## 2. Seed 分区

| 阶段 | Seed | 用途 |
|---|---|---|
| train | 30001-30032 | actor/control 训练与 oracle 数据 |
| selection | 41001-41020 | checkpoint 选择 |
| calibration | 42001-42020 | 规则阈值和实验管线校准 |
| dev | 43001-43030 | 开发期方法比较和消融 |
| final | 70001-70050 | 一次性最终盲测 |

`70001-70050` 禁止用于训练、oracle 标签、阈值调整或 checkpoint 选择。
训练入口和控制器数据加载器会直接拒绝冻结 seed。最终评估还校验
checkpoint 中的训练/验证 seed 血缘；血缘缺失的 checkpoint 不允许进入
final。

## 3. 预注册矩阵

默认场景：

```text
wave_interval ∈ {47.5, 52.5, 57.5, 62.5, 67.5}
profile       ∈ {none, compound_light, compound_medium, compound_heavy}
budget_ms     ∈ {10, 50, 200}
waves         = 12
```

正式基线至少包含：

- `heuristic`
- `cp_sat`
- `cp_sat_repair`
- `event_cp_sat_repair`
- `rl`
- `rl_cp_sat` rule controller
- `rl_cp_sat` learned controller

所有超参数必须在 selection/dev 阶段冻结后再进入 final。

## 4. Manifest

`run_publishable_benchmark.py` 为每次实验写入 `manifest.json`，内容包括：

- protocol version、phase 和 seed 列表；
- 完整环境配置、profile/load/budget/solver 矩阵；
- Git commit、dirty status hash、diff hash；
- NumPy、OR-Tools、PyTorch 版本；
- actor/control checkpoint 路径、SHA-256 和 seed 血缘。

`scenario_id` 绑定：

```text
profile + load + waves + seed + full config hash
+ resolved disruption schedule hash + scenario tape version
```

solver 不进入 `scenario_id`，因此同场景不同 solver 必须共享该 ID。

## 5. 输出契约

### runs.csv

主键为 `scenario_id + solver`，保存：

- 完成/错失架次、SGR、最差波次；
- runtime、p50/p95/p99/max 全决策延迟和 solve-call 延迟；
- solve calls、deadline misses、fallbacks；
- sortie-loss area、resilience index；
- 恢复时间和右删失标记；
- manifest/config/schedule/tape hash。

### waves.csv

主键为 `run_key + wave_index`，保存目标、启动、完成和错失架次。该表用于
恢复曲线和缺额面积，不作为统计独立样本。

### decisions.csv

主键为 `run_key + decision_index`。所有 solver 均记录端到端
`choose_action()` 延迟；repair solver 额外记录控制特征、trigger、scope、
budget、neighborhood、CP 状态、solve time、deadline miss 和 action source。

## 6. 指标定义

### 主指标

```text
N_sortie(H) = simulation horizon 内完成的 launch_done 数量
```

### 架次损失与韧性

同 solver、seed、load 的 `none` 为配对基线：

```text
loss_wave = max(0, sorties_none_wave - sorties_disrupted_wave)
loss_area = Σ loss_wave
resilience = 1 - loss_area / Σ sorties_none_wave
```

### 恢复时间

每个复合扰动窗口结束后，首次满足以下条件的波次视为恢复：

```text
sorties_disrupted >= sorties_none - 1
```

且必须连续保持两波。episode 结束前未满足的样本标记
`recovery_censored=1`，不混入普通恢复时间均值。

### 尾部风险

- 完成架次 lower-tail CVaR10；
- `loss_area` 最大 10% 样本的 upper-tail CVaR90；
- 最差波次架次。

### 实时性

- 决策 latency p50/p95/p99/max；
- solve-call latency p50/p95/p99/max；
- `deadline_miss_rate = misses / solve_calls`；
- fallback rate；
- 相同质量下的最小计算预算。

## 7. 统计分析

对每个 candidate/reference：

1. 在每个 seed 内对预注册 profile/load 矩阵求平均；
2. 计算 seed-block paired mean delta 和 95% bootstrap CI；
3. 报告 win/tie/loss；
4. 使用 exact two-sided sign test；
5. 多基线比较使用 Holm 校正。

禁止把多个波次或同一 seed 下的多个 load/profile 当成独立样本。

建议的 final 成功门槛需在 final 运行前冻结，例如：

```text
平均提升 >= 1.0 架
win rate >= 70%
Holm-adjusted p < 0.05
p95 latency 满足目标预算
```

## 8. 运行命令

开发集小矩阵：

```bash
python -m scripts.run_publishable_benchmark \
  --phase dev \
  --seeds 43001 43002 43003 \
  --profiles none compound_medium compound_heavy \
  --intervals 52.5 57.5 62.5 \
  --budgets-ms 10 50 200 \
  --solvers heuristic cp_sat_repair event_cp_sat_repair rl_cp_sat \
  --workers 3 \
  --output-dir outputs/publishable_dev
```

配对分析：

```bash
python -m scripts.analyze_publishable_benchmark \
  outputs/publishable_dev/runs.csv \
  --candidate rl_cp_sat_b50 \
  --references heuristic cp_sat_repair_b50 \
  --summary-output outputs/publishable_dev/summary.csv \
  --comparison-output outputs/publishable_dev/comparisons.csv
```

Oracle 数据：

```bash
python -m scripts.collect_repair_control_oracle \
  --phase train \
  --profiles compound_light compound_medium compound_heavy \
  --intervals 47.5 57.5 67.5 \
  --output outputs/repair_control_train.csv
```

控制器训练：

```bash
python -m scripts.train_repair_control \
  --train-csv outputs/repair_control_train.csv \
  --validation-csv outputs/repair_control_selection.csv \
  --checkpoint checkpoints/repair_control.pt
```

## 9. 解释边界

- 10/50/200 ms 是端到端 soft deadline。OR-Tools 设置剩余求解时间，但
  当前没有进程级硬中断；超时解不执行。
- soft deadline 使临界预算下的 fallback 路径受机器抖动影响；正式实验
  必须固定硬件、进程数和后台负载，并保留 manifest 中的运行环境信息。
- Manifest 可以证明代码、依赖和模型内容，但 dirty worktree 的结果应在
  commit 后重跑，才能作为最终论文证据。
- 单 seed 结果只用于 smoke/debug，不报告显著性。
