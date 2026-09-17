# CBS 固定批次路径规划

## 目标与边界

当前 Cooperative A* 按调度顺序逐个规划路径，先规划的实体成为后续
实体的固定障碍。CBS 接口用于联合规划一组已知起点、终点和释放时间的
移动请求，目标为最小化路径持续时间之和：

```text
SOC = sum(route.end_time - route.start_time)
```

在低层搜索时域有限、未设置 CBS 展开上限且存在解的条件下，返回结果对
该固定请求集合具有 SOC 最优性。该保证不覆盖任务分配、目标选择、随机
作业时间或整个多波次调度。

## 算法

1. 根节点为各实体忽略同批其他实体时的最短时空路径。
2. 检测最早的节点冲突、同边重叠或对向边冲突。
3. 分别为冲突双方增加节点或有向边时刻约束。
4. 每个分支仅重新规划被约束实体。
5. 约束树按 SOC、最大到达时间和约束数排序。
6. 第一个无冲突节点作为最优批次方案返回。

既有 `SpaceTimeTrafficPlanner` 预约被视为不可修改的动态障碍。批次内
所有路径求解成功后才统一写入预约表，失败时不会提交部分路径。

## 接口

```python
from env.cbs_planner import RouteRequest

plan = traffic_planner.plan_batch(
    [
        RouteRequest(
            entity_id="tractor_0",
            source="deck_pathway_middle_3",
            target="deck_parking_42",
            start_time=0.0,
            shared_nodes=(
                "deck_pathway_middle_3",
                "deck_parking_42",
            ),
        ),
    ]
)
```

`CBSPlan` 返回：

- `routes`：各实体的无冲突时空路径；
- `sum_of_costs`：路径持续时间之和；
- `makespan`：最晚到达时间；
- `expanded_nodes`：CBS 高层展开节点数。

## 当前对照

运行：

```bash
python -m scripts.benchmark_cbs_paths
```

45 机位项目甲板图上的固定双任务瓶颈案例：

| 规划器 | SOC | Makespan |
| --- | ---: | ---: |
| 顺序 Cooperative A* | 11 | 8 |
| CBS | 10 | 6 |

CBS 让第一个实体等待并改走另一条通道，为第二个实体释放瓶颈，总路径
持续时间减少 1 分钟。

## RL 执行链接入

`--cbs-replan` 启用后，环境保留原三层 RL 动作，不让策略直接选择道路。
每新增一个同刻移动任务，环境释放该时刻尚未执行的保障车和载机牵引
预约，并用 CBS 重新规划整批路径。完成事件、作业剩余时间和甲板路径
记录随新路径同步更新。

为隔离 CBS 的真实收益，只有新方案 SOC 严格低于原方案时才替换路径；
求解超出 `--cbs-max-expanded-nodes` 或无解时恢复原 Cooperative A*
预约。牵引车到达飞机前的接近路径暂不重排。

复现实验：

```bash
./scripts/run_rl_cbs_experiment.sh
```

初步结果表明，CBS 在人工瓶颈案例中可以降低 SOC，但当前 RL 实际生成的
移动批次已经达到相同 SOC。固定 60 分钟场景 `seed=10007` 中，RL 开关
CBS 均完成 120 架次，266 次批量求解中没有严格改善。47.5 分钟高负载
的 10 个新种子上，两者均值均为 101.00；2104 次批量求解中同样没有
严格 SOC 改善。因此当前 checkpoint 不因 CBS 获得架次提升。
固定场景单次本地墙钟时间由约 28.1 秒增至 31.4 秒，该单次计时只用于
说明额外搜索存在开销，不作为正式性能统计。

这说明当前瓶颈主要不在“同刻固定任务的路径组合”，下一步若继续研究
路径—调度协同，应把任务顺序、车辆分配或目标选择纳入联合优化，而不是
仅在固定任务后重排等成本路径。
