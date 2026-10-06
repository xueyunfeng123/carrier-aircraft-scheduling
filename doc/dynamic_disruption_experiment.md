# 动态扰动实验

## 环境定义

扰动使用独立随机流 `seed + 1_000_003`，不改变作业时长、弹药量等基础随机过程。同一 seed 下各算法共享相同扰动日程。

扰动开始后立即可观测：

- 车辆故障：故障车辆不能接收新任务，已开始任务继续完成。
- 跑道关闭：关闭跑道不能接收新放飞任务。
- 飞机停飞：停飞飞机不能开始保障或放飞，但允许安全回收。
- 服务降速：仅影响降速期间新开始的加油、检修和装弹作业。

预设包含 3 个同步冲击窗口，较重档位包含较轻档位的故障目标。

| 档位 | 故障车辆 | 关闭跑道 | 停飞飞机 | 服务倍率 | 单窗口时长 |
|---|---:|---:|---:|---:|---:|
| light | 1 | 1 | 2 | 1.25 | 10-20 min |
| medium | 3 | 1 | 5 | 1.50 | 25-40 min |
| heavy | 6 | 2 | 9 | 2.00 | 40-60 min |

未来扰动不会通过事件队列提前泄露。RL 观测维度和 schema 版本保持不变；当前车辆/跑道可用性通过目标掩码表示，服务降速通过已有资源容量特征表示。

## 探索性结果

设置：60 min 波次、12 波、seed `65001-65003`。比较 Heuristic、滚动 CP-SAT、固定负载 RL 和跨负载 RL。表中为平均完成架次。

| 扰动 | Heuristic | CP-SAT | 固定 RL | 跨负载 RL |
|---|---:|---:|---:|---:|
| none | 118.00 | 118.67 | 119.67 | 120.00 |
| light | 115.67 | 119.00 | 120.00 | 119.67 |
| medium | 116.33 | 119.33 | 120.00 | 120.33 |
| heavy | 108.00 | 115.00 | 115.00 | 114.33 |

heavy 相对各自无扰动均值的下降量：

- Heuristic：10.00
- CP-SAT：3.67
- 固定 RL：4.67
- 跨负载 RL：5.67

## 结论

动态扰动形成了有效实验压力，现有 RL 在 heavy 下平均领先 Heuristic 6.33-7.00 架次，但未稳定超过 CP-SAT。固定 RL 在三个 heavy seed 上相对 CP-SAT 为 `+1/+1/-2`，跨负载 RL 为 `-1/-1/0`。

因此当前结果只能支持“RL 相对规则启发式具有更好的动态调度表现”，不能支持“RL 在动态扰动下显著优于 CP-SAT”。动态环境属于实验问题贡献，不应直接表述为算法创新。

下一步应在不改网络结构的前提下进行 disruption-domain randomization：训练时混合 none/light/medium/heavy，并按扰动后恢复速度、窗口缺额和尾部风险选择 checkpoint。若扩展测试后仍不能稳定超过 CP-SAT，再考虑加入事件历史编码或时序图网络。

## 扰动域随机化筛选设计

`train_rl` 支持训练与验证各自配置多个扰动档位和多个波次间隔，并对两者取笛卡尔积。PPO rollout 以固定顺序均衡轮转每个场景单元。checkpoint 使用以下字典序目标，避免高负载场景的高均值掩盖困难场景退化：

1. 最大化最差 `profile × wave_interval` 单元在验证 seed 上的平均完成架次。
2. 最大化全部验证场景的平均完成架次。
3. 最大化最差单局完成架次。
4. 最小化全部验证场景的平均缺额。

筛选实验从 `checkpoints/rl_multiload_bc.pt` 初始化，训练 seed 为 `30001-30005`，筛选 seed 固定为 `41001-41005`，负载为 50/70 分钟，扰动为 none/light/medium/heavy。8 次 PPO 更新恰好覆盖一轮 8 个场景单元。禁止将 `70001-70050` 用于本实验的训练、筛选或报告。

预注册的工程筛选门槛如下：

- randomized RL 的 heavy 汇总均值严格高于 Heuristic、滚动 CP-SAT、固定负载 RL 和跨负载 RL。
- randomized RL 的综合均值不低于上述四种对照中的最佳值。
- randomized RL 的最差场景单元均值不低于初始化 checkpoint。

五个筛选 seed 只用于选择是否进入扩大实验，不能满足项目正式准入条件。正式结论仍要求至少 30 个全新冻结 seed，并对主要对照达到配对显著性 `p < 0.05`。

## 复现

```bash
python -m scripts.benchmark_dynamic_disruptions \
  --runs 3 \
  --seed 65001 \
  --waves 12

# 均衡扰动域随机化 PPO 与 4 类对照的多负载筛选
./scripts/run_rl_disruption_randomization_experiment.sh
```

明细与汇总：

- `outputs/dynamic_disruption_headroom_detail.csv`
- `outputs/dynamic_disruption_headroom.csv`
