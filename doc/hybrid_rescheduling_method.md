# 复合扰动下风险感知、事件切换 RL-CP-SAT 重调度

## 1. 研究问题

固定时域内最大化完成放飞架次：

\[
\max_\pi \ \mathbb{E}[N_{\mathrm{sortie}}(H)]
\]

错失架次、韧性、恢复时间和决策延迟均为评估指标，不进入主目标。
方法面向车辆故障、跑道关闭、保障减速、临时停飞和实体飞机故障同时
出现的在线场景，而不是静态实例上的一次性离线优化。

## 2. 投稿创新主线

1. **反事实一致的复合扰动 DES**  
   `ScenarioTape` 按语义事件键生成随机量，使不同 solver 在动作顺序
   不同的情况下仍共享同一随机场景。`compound_*` 场景进一步加入飞机
   故障、机库转运、维修、备机替换和物理机库存守恒。

2. **风险感知的事件切换混合控制**
   RL actor 负责低压力名义流；扰动激活或故障生命周期未清空时切换到
   CP-SAT。episode 开始时再根据已知波次负载与场景风险等级决定是否全程
   使用 CP-SAT。控制器不读取扰动实现、未来事件时刻或 seed。

3. **带 incumbent 和安全回退的滚动局部修复**  
   CP-SAT 使用期望作业时长、资源累计约束、在执行作业的剩余占用区间、
   incumbent hints 和 actor guidance。超出墙钟预算、无可行解或动作失效
   时立即回退到 actor/heuristic，环境不会执行非法动作。

4. **质量-延迟联合评估**  
   每次决策记录真实端到端墙钟时间、CP 求解时间、预算、邻域、解状态、
   deadline miss 和 fallback。主结果按 seed 配对，报告架次差、尾延迟、
   CVaR、韧性和恢复时间。实验中的 10/50/200 ms 是 CP 求解 compute cap，
   不是端到端硬 deadline。

## 3. 复合扰动环境

新场景不改变历史 `none/light/medium/heavy`：

| 场景 | 车辆故障 | 跑道关闭 | 临时停飞 | 实体故障 | 减速 |
|---|---:|---:|---:|---:|---:|
| `compound_light` | 1 | 1 | 1 | 1 | 1.25 |
| `compound_medium` | 3 | 1 | 3 | 2 | 1.50 |
| `compound_heavy` | 6 | 2 | 5 | 4 | 2.00 |

每个 episode 在 25%、50%、75% 时域附近产生三个冲击窗口。同一窗口
内 hold/failure 目标互斥；复合窗口不重叠；短时域下最大持续时间限制为
时域的 20%，防止所有事件被压到 `t=0`。

实体故障按以下生命周期推进：

```text
故障检测 -> 安全回收（若在空中）-> 转运机库 -> 维修
                                      |
                                      +-> 备机转运上甲板
维修完成 -> hangar-ready，可在后续故障中再次作为替换机
```

库存不变量要求每个 physical aircraft ID 恰好处于 deck、to-hangar、
under-repair、hangar-ready 或 to-deck 中的一类。

## 4. 滚动 CP-SAT 修复

规划候选为当前合法的加油、检查和装弹作业。模型包括：

- 加油车、检查车、装弹车累计容量；
- 弹药转运车、上下层升降机分阶段容量；
- 可选的人员需求容量；
- 正在执行作业到其完成事件之间的固定占用区间；
- 当前波截止时间前完成保障的高权重目标；
- incumbent 启动时间偏差惩罚；
- actor action-pair score 形成的次级优先级。

`scope` 的语义：

- `current_wave`：邻域不超过本波剩余任务席位；
- `two_waves`：覆盖本波剩余席位和下一波需求；
- `affected`：优先保留与当前车辆、服务、跑道或飞机扰动直接关联的作业。

## 5. 最终风险门控

最终方法 `risk_aware_rl_cp` 冻结为以下单调规则：

```text
if wave_interval <= 47.5:
    use CP-SAT for the complete episode
elif risk_multiplier >= 2.0 and wave_interval <= 67.5:
    use CP-SAT for the complete episode
elif risk_multiplier >= 1.5 and wave_interval <= 57.5:
    use CP-SAT for the complete episode
elif disruption active or failure lifecycle not cleared:
    use CP-SAT
else:
    use the RL actor
```

其中 `compound_light/medium/heavy` 的风险倍率分别为
`1.25/1.50/2.00`。阈值只在 dev seeds `43001-43030` 上确定；final
seeds `70001-70050` 未用于规则、checkpoint 或超参数选择。

RL actor 使用扰动域随机化 PPO checkpoint：

```text
checkpoints/rl_disruption_randomized_ppo.pt
SHA-256 dd8e93d44caac52036b3d869a2a23f45e1fa2b8b31ea114fd03c66313e7d9b5f
train seeds      30001-30005
selection seeds  41001-41005
```

## 6. 探索性学习控制器

在线状态为 13 维，仅使用当前可观测信息：

- 归一化时间、波次进度、截止时间；
- 待回收比例、候选动作比例；
- 活跃扰动、不可用车辆/跑道/飞机比例；
- 服务减速与剩余容量；
- 故障阻塞槽位比例。

四个离散 head 输出：

```text
trigger          ∈ {off, on}
scope            ∈ {current_wave, two_waves, affected}
budget_ms        ∈ {10, 50, 200}
neighborhood     ∈ {8, 16, 32}
```

训练数据由 `scripts.collect_repair_control_oracle` 生成。每个真实控制调用
点冻结环境快照，并在相同 scenario tape 上比较：

- 关闭修复；
- `3 budgets × 3 neighborhoods`，固定 `scope=two_waves` 的九个修复候选。

每个候选向前滚动两波，按以下字典序选择标签：

```text
完成架次 -> -错失架次 -> -deadline miss -> -运行时
```

完全无质量差异的样本降低权重。训练脚本
`scripts.train_repair_control` 对 trigger 使用全样本交叉熵；其余 head
只在 oracle 触发时计入损失。

该四 head 控制器属于并行探索，不是 final 主方法。已有 oracle/消融没有
证明它稳定优于强 CP-SAT，因此不得把下述结构写成 final 已验证贡献。

## 7. 在线算法

```text
observe current state
evaluate the frozen pressure/risk gate
if the gate forces CP-SAT:
    execute the legal CP-SAT action
elif a disruption is active or a failed-aircraft lifecycle is unresolved:
    execute the legal CP-SAT action
else:
    execute the legal RL actor action
record source and end-to-end choose_action latency
```

## 8. Final 盲测结果

冻结 commit `250bc33` 在 final seeds `70001-70050` 上执行 1000 个场景、
5000 个 paired runs。每个场景为 12 波，空间图开启，compute cap 为
50 ms。

| 方法 | 平均完成架次 | CVaR10 | 全决策 P95 | 平均运行时间 |
|---|---:|---:|---:|---:|
| Heuristic | 109.969 | 87.09 | 180.50 ms | 109.29 s |
| RL | 113.100 | 91.99 | 194.19 ms | 120.81 s |
| Adaptive RL-CP | 113.932 | 94.81 | 188.68 ms | 119.37 s |
| CP-SAT | 114.046 | 95.60 | 185.23 ms | 117.86 s |
| Risk-aware RL-CP | **114.157** | **95.60** | 187.91 ms | 119.36 s |

Risk-aware RL-CP 相对 CP-SAT：

| 范围 | 平均差 | 95% seed-block CI | W/T/L | sign test | Holm |
|---|---:|---:|---:|---:|---:|
| dynamic-only | +0.1107 | [0.0613, 0.1600] | 35/2/13 | 0.00209 | - |
| all | +0.1110 | [0.0570, 0.1660] | 35/5/10 | 0.000247 | 0.00247 |
| compound light | +0.1920 | [0.0760, 0.3120] | 29/10/11 | 0.00643 | 0.03856 |
| compound medium | +0.1400 | [0.0680, 0.2120] | 27/13/10 | 0.00763 | 0.03856 |
| compound heavy | 0.0000 | [0.0000, 0.0000] | 0/50/0 | 1.0 | 1.0 |

相对 RL 和 Heuristic 的 dynamic-only 平均提升分别为 `+1.3333` 和
`+4.6507` 架。原始证据位于远端
`outputs/final_main_50/`，manifest hash 为
`16e50dd4f5970d2da5624b37ff2db79ce8043915b4b66354150e84591bb6e92e`。

该结果证明统计优势，但没有达到此前建议的 `>=1.0` 架实用效应门槛；
也没有质量-延迟 Pareto 优势。论文不得写成“大幅超过 CP-SAT”或“严格
实时 50 ms”，应写成小效应、可重复的平均吞吐提升，且 CVaR10 持平。

## 9. 后续实验

1. 主比较：Heuristic、旧 rolling CP-SAT、always-on CP repair、
   event-triggered CP repair、RL actor、rule-triggered hybrid、
   learned hybrid。
2. 消融：无 scenario tape、无 actor guidance、always-on repair、
   固定预算、固定邻域、无 incumbent、无 active-occupancy。
3. 泛化：未见过的负载、扰动强度、资源容量和故障持续时间。
4. 质量-延迟：10/50/200 ms 下架次、p95/p99、deadline miss、
   fallback。
5. 韧性：none 配对缺额面积、恢复时间、worst-10% sortie-loss CVaR。

## 10. 限制

- 当前预算包含模型构建和求解，但没有通过独立 worker 强制终止超时
  调用；超时结果被丢弃并回退，因此属于 soft real-time。
- 必须同时报告全决策延迟和 solve-call 专用 p50/p95/p99/max；前者会被
  大量快速 actor/fallback 决策稀释。
- CP-SAT 使用期望作业时长，不是随机时长的机会约束模型。
- ready reserve aircraft 可能快速替换未完成保障的故障机，因此严重度
  与架次损失不保证逐样本单调。
- `affected` 是基于当前扰动类型的可解释筛选，不是学习得到的图邻域。
- final 的 CP compute cap 为 50 ms，但模型构建、动作重验证和环境侧规划
  使端到端 P95 达到约 188 ms。
- 相对 CP-SAT 的 final 平均增益仅 0.111 架，统计显著不等于工程显著。
- lower-tail CVaR10 与 CP-SAT 持平；resilience ratio 略低，不能主张
  所有韧性指标均改善。
