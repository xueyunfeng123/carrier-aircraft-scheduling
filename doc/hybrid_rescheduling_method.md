# 复合扰动下事件触发、预算自适应 RL-CP-SAT 重调度

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

2. **事件触发的分层混合控制**  
   RL actor 给出保障作业与飞机的优先分数；控制器只在波次或扰动状态
   变化等关键事件选择是否调用 CP-SAT，并联合选择修复范围、求解预算和
   邻域规模。该设计将学习重点从“替代优化器”转为“决定何时、以多大
   计算代价调用优化器”。

3. **带 incumbent 和安全回退的滚动局部修复**  
   CP-SAT 使用期望作业时长、资源累计约束、在执行作业的剩余占用区间、
   incumbent hints 和 actor guidance。超出墙钟预算、无可行解或动作失效
   时立即回退到 actor/heuristic，环境不会执行非法动作。

4. **质量-延迟联合评估**  
   每次决策记录真实端到端墙钟时间、CP 求解时间、预算、邻域、解状态、
   deadline miss 和 fallback。主结果按 seed 配对，报告架次差、尾延迟、
   CVaR、韧性和恢复时间。

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

## 5. 学习控制器

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

## 6. 在线算法

```text
observe state
actor proposes fallback action and action-pair scores
if action is launch/recovery:
    execute immediately
else:
    controller selects trigger/scope/budget/neighborhood
    if trigger:
        solve rolling CP-SAT repair under wall-clock budget
        if feasible and within deadline:
            revalidate and execute first repaired action
    execute legal fallback action
```

## 7. 已完成的开发证据

开发 smoke 使用 seed `43001`、57.5 分钟波次、4 波、关闭空间图。结果只
证明链路可运行，不构成统计结论：

| 方法 | `none` | `compound_heavy` | heavy 全决策 p95 |
|---|---:|---:|---:|
| Heuristic | 80 | 77 | 0.05 ms |
| always-on CP repair | 80 | 74-78 | 约 10-40 ms |
| event-triggered CP repair | 80 | 77 | 约 0.2 ms |
| rule-triggered RL-CP | 80 | 76 | 约 2 ms |

当前结论：

- always-on CP repair 在重复 smoke 中受 soft deadline 抖动影响，10 ms
  heavy 结果为 74-78 架；混合方法固定为 76 架；
- 混合方法比 50/200 ms CP repair 多 2 架，但 10 ms 下可相差 -2 至 +2；
- 非学习 event-triggered CP 达到 77 架，与 Heuristic 持平，并比当前
  RL-CP 多 1 架；现有静态 actor guidance 对动态故障存在负迁移；
- RL-CP 仍比 Heuristic 少 1 架，尚未达到论文性能主张；
- 三档预算架次相同，因为规则控制器只触发少量修复调用；
- 必须完成 oracle 数据采集和学习控制器训练，再进行正式开发集比较。

全决策 p95 会被快速 fallback 稀释。该 smoke 中 event-triggered CP 的
solve-call p95 随预算约为 9/50/199 ms，RL-CP 约为 10/49/49 ms；正式
表格必须同时报告两种延迟口径。

原始证据位于 `outputs/publishable_representative/`。

## 8. 必做实验

1. 主比较：Heuristic、旧 rolling CP-SAT、always-on CP repair、
   event-triggered CP repair、RL actor、rule-triggered hybrid、
   learned hybrid。
2. 消融：无 scenario tape、无 actor guidance、always-on repair、
   固定预算、固定邻域、无 incumbent、无 active-occupancy。
3. 泛化：未见过的负载、扰动强度、资源容量和故障持续时间。
4. 质量-延迟：10/50/200 ms 下架次、p95/p99、deadline miss、
   fallback。
5. 韧性：none 配对缺额面积、恢复时间、worst-10% sortie-loss CVaR。

## 9. 限制

- 当前预算包含模型构建和求解，但没有通过独立 worker 强制终止超时
  调用；超时结果被丢弃并回退，因此属于 soft real-time。
- 必须同时报告全决策延迟和 solve-call 专用 p50/p95/p99/max；前者会被
  大量快速 actor/fallback 决策稀释。
- CP-SAT 使用期望作业时长，不是随机时长的机会约束模型。
- ready reserve aircraft 可能快速替换未完成保障的故障机，因此严重度
  与架次损失不保证逐样本单调。
- `affected` 是基于当前扰动类型的可解释筛选，不是学习得到的图邻域。
- 当前代表性结果尚未证明 learned hybrid 超过最强 Heuristic。
