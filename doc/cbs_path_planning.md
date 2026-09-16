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

## 后续接入

主仿真当前每收到一个调度动作就立即预约路径，尚不能提前看到同一时刻
后续将启动的移动任务。下一阶段需要增加移动请求批处理：先锁定资源和
目标，再将同一决策时刻的牵引及保障移动统一提交给 CBS，最后创建完成
事件。完成该接入后才能比较完整波次的架次、道路等待和求解耗时。
