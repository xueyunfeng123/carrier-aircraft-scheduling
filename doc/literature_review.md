# 舰载机甲板作业调度与强化学习求解：系统文献调研

> 检索与核验日期：2026-09-03
>
> 主题范围：舰载机回收、保障与放飞的资源受限动态调度；随机扰动；离散事件仿真；强化学习、层次强化学习与模仿学习。
>
> 项目目标：在固定时域内最大化成功放飞架次。
>
> 飞书文档：<https://mcnubpcdg10x.feishu.cn/docx/Pt6XdbkWNo0ja6xu6NXcDYySnUg>

## 1. 检索方法与真实性核验

### 1.1 纳入和排除标准

纳入标准：

1. 直接研究航母飞行甲板、舰载机出动、回收、保障、机位、资源或路径协同调度；
2. 或者研究与本项目结构高度同构的随机资源受限项目调度、动态作业车间调度、离散事件仿真优化；
3. 或者对事件驱动决策、层次/多动作策略、行为克隆、PPO、非法动作掩码提供可迁移的方法依据；
4. 标题、作者、年份、来源和 DOI/arXiv/正式页面至少由两个元数据来源交叉核对，或由出版社正式页面直接核对。

排除标准：

- 仅研究舰载机飞行动力学、着舰控制或一般航迹规划，且不含任务/资源调度；
- 仅研究民航机场跑道排序，缺少与甲板资源保障或循环出动的实质联系；
- 只有二手转载、无法核对作者或出版信息；
- 与调度关系弱，仅因标题中出现 aircraft、carrier、PPO 等词而命中。

### 1.2 检索关键词

舰载机直接相关：

```text
"aircraft carrier" "flight deck" scheduling
"carrier aircraft" launch recovery scheduling
"carrier-based aircraft" support operation scheduling
"cyclic flight deck operations" sortie generation
"carrier aircraft" dynamic scheduling rolling horizon
"carrier aircraft" reinforcement learning scheduling
"carrier-borne aircraft" hierarchical reinforcement learning
"aircraft sortie generation" queueing network
```

随机与动态资源调度：

```text
"stochastic resource-constrained project scheduling"
"dynamic stochastic resource-constrained multi-project scheduling"
"proactive reactive scheduling" resource constrained
"dynamic rescheduling" machine failure resource disruption
"resource transfer" multi-project scheduling
```

离散事件仿真与学习调度：

```text
"discrete event simulation" scheduling optimization
"simulation-based approximate dynamic programming" scheduling
"event-based reinforcement learning" job shop
"reinforcement learning" job shop scheduling graph neural network
"hierarchical reinforcement learning" dynamic scheduling
"multi-action PPO" flexible job shop
"behavior cloning" job shop scheduling
"invalid action masking" policy gradient
```

### 1.3 数据库检索记录

| 数据库/平台                              | 实际检索方式                              | 主要用途                  | 结果处理                                             |
| ----------------------------------- | ----------------------------------- | --------------------- | ------------------------------------------------ |
| OpenAlex                            | 4 组主题检索，每组返回 25--30 条，共取得 105 条原始记录 | 构建候选池、查引用量级、发现相邻论文    | 按 DOI/标题去重；剔除材料、控制等误命中                           |
| Semantic Scholar                    | 主题检索、精确标题检索、参考文献与被引文献追踪             | 发现舰载机专用 RL、近期论文和引用链   | 不采用其自动摘要作为唯一事实来源                                 |
| Crossref                            | 对最终候选 DOI 逐条查询                      | 核验标题、作者、期刊、卷期、页码和 DOI | 以期刊卷期年为主，online-first 年另行说明                      |
| arXiv                               | 精确题名与 arXiv ID 检索                   | 核对公开稿、版本日期和预印本状态      | 正式发表者标注“正式论文，arXiv 为公开稿”                         |
| IEEE Xplore                         | 精确 DOI/题名页                          | 核验 IEEE 期刊和会议元数据及摘要   | 用正式页面替代聚合站信息                                     |
| ScienceDirect                       | DOI/题名页及 Crossref/OpenAlex 元数据      | 核验 Elsevier 期刊论文      | 遇到访问限制时以 Crossref、OpenAlex、Semantic Scholar 交叉确认 |
| Wiley/Hindawi、Springer、MDPI、SciOpen | DOI 或正式文章页                          | 获取开放摘要、方法、实验和参考文献     | 用于精读和引用追踪                                        |

说明：

- 最终表中的 30 篇均有 DOI 或正式会议页面；没有仅靠搜索摘要拼接出来的条目。
- 表中没有“仅预印本”论文。部分正式论文同时提供 arXiv 公开稿，已明确说明。
- 引用量随数据库和日期变化，且用户未要求引用量字段，因此不将即时引用数写入主表。
- 2025--2026 年部分论文的 DOI 年份早于卷期年。本文统一以正式卷期年作为“年份”，必要时在来源栏标注 online-first。

### 1.4 证据标记

下文使用两种标记：

- **\[文献结论]**：论文摘要、正文、表格或正式出版页面直接支持的内容；
- **\[本项目推断]**：根据多篇文献和当前项目结构作出的迁移判断，不代表原论文已经验证。

## 2. 文献版图概览

**\[文献结论]** 舰载机甲板调度的英文同行评议文献数量明显少于通用 JSSP/FJSP、RCPSP 和机场调度文献，且作者群和发表渠道较集中。公开研究通常分别处理：

- 出动排序、弹射器与滑行路径；
- 甲板保障作业及人员/设备/空间资源分配；
- 随机工时下的鲁棒基线计划；
- 扰动发生后的滚动时域或基线-反应式重调度；
- 近年的多智能体、层次强化学习、安全强化学习和 GNN 调度。

**\[本项目推断]** 现有直接文献已经足以否定“首次使用强化学习解决舰载机调度”这类宽泛创新表述，但仍未形成一个被公开复现的统一范式，同时覆盖“回收—加油/挂弹—放飞”闭环、A/B 多波次、随机时长、共享资源、事件驱动分层动作、BC 预训练和 PPO 微调，并以固定时域成功放飞架次作为唯一目标。

## 3. 高相关论文表

### 3.1 舰载机甲板与出动/回收调度

| ID  | 标题                                                                                                                                                             | 作者                                                                                              |   年份 | 期刊/会议                                                        | DOI/链接                                                                                                               | 研究问题                          | 使用方法                              | 与本项目的关联                                                         | 相关性  |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---: | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------- | --------------------------------------------------------------- | ---- |
| D01 | Analysis of Aircraft Sortie Generation with the Use of a Fork-Join Queueing Network Model                                                                      | Dennis C. Dietz; Richard C. Jenkins                                                             | 1997 | Naval Research Logistics 44(2):153-164                       | [DOI](https://doi.org/10.1002/%28SICI%291520-6750%28199703%2944%3A2%3C153%3A%3AAID-NAV1%3E3.0.CO%3B2-8)              | 舰载机出动生成过程的吞吐与瓶颈               | Fork-join 排队网络                    | 奠定“并行保障后汇合放飞”的早期分析框架；可用于解释加油和挂弹并行后的同步瓶颈                         | 直接相关 |
| D02 | Comparing the Performance of Expert User Heuristics and an Integer Linear Program in Aircraft Carrier Deck Operations                                          | Jason C. Ryan; Ashis G. Banerjee; Mary L. Cummings; Nicholas Roy                                | 2014 | IEEE Transactions on Cybernetics 44(6):761-773               | [DOI](https://doi.org/10.1109/TCYB.2013.2271694)                                                                     | 甲板重规划中专家经验与优化器的比较             | 人机协同决策支持、整数线性规划、专家启发式             | 直接支持 Heuristic 与精确/数学规划基线并列比较；提醒“优化器”不天然胜过专家规则                  | 直接相关 |
| D03 | Dynamic Aircraft Carrier Flight Deck Task Planning based on HTN                                                                                                | Chao Qi; Dan Wang                                                                               | 2016 | IFAC-PapersOnLine 49(12):1608-1613                           | [DOI](https://doi.org/10.1016/j.ifacol.2016.07.810)                                                                  | 新任务到来时的动态甲板任务规划               | Hierarchical Task Network，触发式重规划  | 与事件触发决策、层次任务结构和动态重调度直接对应                                        | 直接相关 |
| D04 | An Extended Flexible Job Shop Scheduling Model for Flight Deck Scheduling with Priority, Parallel Operations, and Sequence Flexibility                         | Lianfei Yu; Cheng Zhu; Jianmai Shi; Weiming Zhang                                               | 2017 | Scientific Programming                                       | [DOI](https://doi.org/10.1155/2017/2463252)                                                                          | 优先级、并行作业和可变工序顺序下的甲板保障调度       | 扩展 FJSP、改进差分进化                    | 与加油/挂弹并行、工序优先关系和多飞机资源竞争高度一致                                     | 直接相关 |
| D05 | A Proactive Robust Scheduling Method for Aircraft Carrier Flight Deck Operations with Stochastic Durations                                                     | Xichao Su; Wei Han; Yu Wu; Yong Zhang; Jie Liu                                                  | 2018 | Complexity                                                   | [DOI](https://doi.org/10.1155/2018/6932985)                                                                          | 随机作业时长下的鲁棒甲板保障计划与资源配置         | OONF 网络、串/并行 SGS、HTLBO、预约束执行策略    | 是本项目随机处理时长、人员/设备/空间共享约束的核心直接参照                                  | 直接相关 |
| D06 | Discrete Event Simulation of Aircraft Sortie Generation on an Aircraft Carrier                                                                                 | Hee Chang Yoon; Seung Heon Oh; Jong Hun Woo; Jung-Hoon Chung; Hyuk Lee; Sun-Ah Jung             | 2023 | Winter Simulation Conference 2023:2439-2449                  | [DOI](https://doi.org/10.1109/WSC60868.2023.10407756)                                                                | 甲板空间、设施、人员和飞机交互下的出动生成过程与出动率   | 航母出动生成离散事件仿真                      | 是少数直接以 sortie generation process/rate 为对象的 DES 论文，与当前仿真环境定位高度一致 | 直接相关 |
| D07 | A Dynamic Scheduling Method for Carrier Aircraft Support Operation under Uncertain Conditions Based on Rolling Horizon Strategy                                | Peilong Yuan; Wei Han; Xichao Su; Jie Liu; Jingyu Song                                          | 2018 | Applied Sciences 8(9):1546                                   | [DOI](https://doi.org/10.3390/app8091546)                                                                            | 不确定条件下人员、站位、保障设备与消耗资源的动态调度    | 整数规划、周期+事件驱动滚动时域、双种群遗传算法          | 与事件驱动重调度和资源状态变化最直接对应                                            | 直接相关 |
| D08 | Integration Design of Sortie Scheduling for Carrier Aircrafts Based on Hybrid Flexible Flowshop                                                                | Jie Liu; Wei Han; Jie Li; Yong Zhang; Xichao Su                                                 | 2020 | IEEE Systems Journal 14(1):1503-1511                         | [DOI](https://doi.org/10.1109/JSYST.2019.2922261)                                                                    | 从机位到弹射起飞的整体出动调度与滑行协同          | 混合柔性流水车间、双层遗传算法、协同轨迹规划            | 提供“作业层+移动层”一体化建模对照；本项目当前只保留抽象转运时间                               | 直接相关 |
| D09 | A Multi-Objective Hyper-Heuristic Framework for Integrated Optimization of Carrier-Based Aircraft Flight Deck Operations Scheduling and Resource Configuration | Rongwei Cui; Wei Han; Xichao Su; Yong Zhang; Fang Guo                                           | 2020 | Aerospace Science and Technology 107:106346                  | [DOI](https://doi.org/10.1016/j.ast.2020.106346)                                                                     | 作业计划与保障资源配置联合优化               | 多目标超启发式、choice function           | 说明调度策略和资源容量可联合研究；本项目应先固定资源做公平算法比较                               | 直接相关 |
| D10 | A Baseline-Reactive Scheduling Method for Carrier-Based Aircraft Maintenance Tasks                                                                             | Yong Zhang; Changjiu Li; Xichao Su; Rongwei Cui; Bing Wan                                       | 2023 | Complex & Intelligent Systems 9(1):367-397（online 2022）      | [DOI](https://doi.org/10.1007/s40747-022-00784-9)                                                                    | 人员、设备/车间、空间和并行能力约束下的维修计划与扰动恢复 | I-NSGA-II、串行 SGS、完全/部分重调度         | 可作为故障后“全量重排 vs 局部修复”的直接对照                                       | 直接相关 |
| D11 | A Heuristic Solution Framework for the Resource-Constrained Multi-Aircraft Scheduling Problem with Transfer of Resources and Aircraft                          | Xichao Su; Rongwei Cui; Changjiu Li; Wei Han; Jie Liu                                           | 2023 | Expert Systems with Applications 228:120430                  | [DOI](https://doi.org/10.1016/j.eswa.2023.120430)                                                                    | 同时考虑飞机和保障资源转移时间的多机调度          | 串/并行调度生成、机位/资源规则、遗传编程             | 补足当前模型中资源移动与机位转移被简化的问题                                          | 直接相关 |
| D12 | Cooperative Carrier Aircraft Support Operation Scheduling via Multi-Agent Reinforcement Learning                                                               | Hongjie Hao; Xueqin Zhang; Yuan Chi; Rongxin Gao; Anke Xie; Mingliang Xu                        | 2023 | IEEE MDM 2023:297-302                                        | [DOI](https://doi.org/10.1109/MDM58254.2023.00055)                                                                   | 受限空间和资源下的多保障主体协同调度            | Dec-POMDP、集中训练分散执行，与 VDN/QMIX 比较  | 是舰载机保障 MARL 的直接先例；与当前单智能体分层策略形成对照                               | 直接相关 |
| D13 | Autonomous Sortie Scheduling for Carrier Aircraft Fleet under Towing Mode                                                                                      | Zhilong Deng; Xuanbo Liu; Yuqi Dou; Xichao Su; Haixu Li; Lei Wang; Xinwei Wang                  | 2025 | Defence Technology 43:1-12（online 2024）                      | [DOI](https://doi.org/10.1016/j.dt.2024.07.011)                                                                      | 牵引模式下资源分配、站位占用与碰撞规避           | 混合流水车间、MILP、轨迹库、混沌初始化 GA          | 说明显式甲板空间会引入紧耦合时空约束；可作为当前简化模型的外部边界                               | 直接相关 |
| D14 | Human Experience-Guided Reinforcement Learning for Carrier-Based Aircraft Support Operation Scheduling                                                         | Xudong Chen; Yizhe Luo; Qihang Sun; Wenxiao Guo; Zhao Jin; Shuo Feng; Yucheng Shi; Mingliang Xu | 2025 | Defence Technology 54:211-224                                | [DOI](https://doi.org/10.1016/j.dt.2025.07.016)                                                                      | 稀疏奖励、高维状态动作和动态约束下的保障调度        | 经验库匹配干预、actor-critic、混合奖励、自适应引导权重 | 与启发式示范预热最接近，但它采用在线经验干预而非 BC→PPO                                 | 直接相关 |
| D15 | Simulation-Based Two-Stage Scheduling Optimization Method for Carrier-Based Aircraft Launch and Departure Operations                                           | Jue Liu; Nengjian Wang                                                                          | 2025 | Entropy 27(7):662                                            | [DOI](https://doi.org/10.3390/e27070662) / [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12293827/)                | 空间路径与站位匹配/出动排序的耦合优化           | O2DES、AAE-SAC 路径规划、LTA-HPSO 排序    | 是“离散事件仿真+学习/群智能”直接舰载机案例，但聚焦出动与路径                                | 直接相关 |
| D16 | Efficient Collaborative Planning for Carrier-Borne Aircraft Dispatch and Recovery via Hierarchical Reinforcement Learning                                      | Shaohui Zhang; Qiuying Han; Qingshun Wu; Di Zhang; Yafei Li; Mingliang Xu                       | 2026 | Chinese Journal of Aeronautics 39(4):103681（online 2025）     | [DOI](https://doi.org/10.1016/j.cja.2025.103681) / [开放页面](https://www.sciopen.com/article/10.1016/j.cja.2025.103681) | 多波次飞机的流程、站位与资源协同规划            | 三层 HRL、Dec-POMDP、层间通信             | 是“多波次+分层动作”的最直接先例；本项目需突出不同的动作语义与目标函数                            | 直接相关 |
| D17 | Real-Time Sequencing Problem for Aircraft Recovery on Carriers: A Safe Deep Reinforcement Learning Approach                                                    | Changjiu Li; Yong Zhang; Wei Han; Fang Guo; Xinwei Wang; Xichao Su                              | 2026 | Chinese Journal of Aeronautics 39(9):103911（online 2025）     | [DOI](https://doi.org/10.1016/j.cja.2025.103911)                                                                     | 复飞概率、空中等待与安全约束下的实时回收排序        | 安全深度强化学习                          | 对回收时限、风险约束和实时排序有直接启发，但不覆盖保障闭环                                   | 直接相关 |
| D18 | Deep Reinforcement Learning for Carrier-Based Aircraft Flight Deck Operations Scheduling Problem                                                               | Changjiu Li; Wei Han; Haixu Li; Jie Liu; Xinwei Wang; Yong Zhang; Xichao Su                     | 2026 | Neural Networks 200:108776                                   | [DOI](https://doi.org/10.1016/j.neunet.2026.108776)                                                                  | 甲板作业调度中求解质量与实时计算的权衡           | MDP、GNN、深度强化学习                    | 是最接近“状态到甲板调度动作”的正式期刊研究之一                                        | 直接相关 |
| D19 | End-to-End Scheduling for Carrier-Based Aircraft Sortie Operations Using Deep Reinforcement Learning                                                           | Changjiu Li; Wei Han; Yong Zhang; Xinwei Wang; Xichao Su                                        | 2026 | Information Processing & Management 63(4):104590             | [DOI](https://doi.org/10.1016/j.ipm.2025.104590)                                                                     | 舰载机出动过程的端到端调度                 | 深度强化学习                            | 证明“端到端舰载机出动 RL”已是现有研究主题，不能作为本项目的宽泛首创点                           | 直接相关 |
| D20 | Collaborative Scheduling Problem Pertaining to Launch and Recovery Operations for Carrier Aircraft                                                             | Fang Guo; Wei Han; Yujie Liu; Xichao Su; Jie Liu; Changjiu Li                                   | 2026 | Journal of Systems Engineering and Electronics 37(1):287-306 | [DOI](https://doi.org/10.23919/JSEE.2026.000043) / [全文页](https://www.jseepub.com/EN/10.23919/JSEE.2026.000043)       | 起飞与回收并行时的冲突协调                 | 多种群自适应差分进化 MPSADE                 | 与共享起降区、空中等待和放飞延迟的权衡直接相关                                         | 直接相关 |

### 3.2 随机调度、离散事件仿真与学习方法

| ID  | 标题                                                                                                                                          | 作者                                                                                   |   年份 | 期刊/会议                                                                    | DOI/arXiv/链接                                                                                                                                                             | 研究问题                       | 使用方法                                | 与本项目的关联                                   | 相关性  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---: | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- | ----------------------------------- | ----------------------------------------- | ---- |
| M01 | Solving Stochastic Resource-Constrained Project Scheduling Problems by Closed-Loop Approximate Dynamic Programming                          | Haitao Li; Norman K. Womer                                                           | 2015 | European Journal of Operational Research 246(1):20-33                    | [DOI](https://doi.org/10.1016/j.ejor.2015.04.015)                                                                                                                        | 随机工期和资源约束下的闭环自适应策略         | Rollout ADP、CP 增强基策略、前看/回看混合近似      | 为“根据已实现随机时长在线决策”提供经典闭环基线                  | 方法参考 |
| M02 | Simulation Optimization: A Review of Algorithms and Applications                                                                            | Satyajith Amaran; Nikolaos V. Sahinidis; Bikram Sharda; Scott J. Bury                | 2016 | Annals of Operations Research 240(1):351-380                             | [DOI](https://doi.org/10.1007/s10479-015-2019-x)                                                                                                                         | 随机黑箱仿真的优化方法分类              | 随机搜索、元模型、梯度估计、离散/连续仿真优化综述           | 为“仿真器是评价器、求解器是外层决策器”的方法论定位提供依据            | 方法参考 |
| M03 | A Simulation-Based Approximate Dynamic Programming Approach to Dynamic and Stochastic Resource-Constrained Multi-Project Scheduling Problem | Uğur Satıç; Peter Jacko; Christopher Kirkbride                                       | 2024 | European Journal of Operational Research 315(2):454-469（online 2023）     | [DOI](https://doi.org/10.1016/j.ejor.2023.10.046)                                                                                                                        | 随机项目到达、随机工期和多类资源下的长期收益调度   | 无限时域 MDP、Monte Carlo、线性价值近似、ADP     | 与多波次到达和事件时点在线派工同构；目标需改为有限时域架次             | 邻近问题 |
| M04 | Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning                                                                | Cong Zhang; Wen Song; Zhiguang Cao; Jie Zhang; Puay Siew Tan; Chi Xu                 | 2020 | NeurIPS 33:1621-1632                                                     | [正式论文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html) / [arXiv:2010.12367](https://arxiv.org/abs/2010.12367) | 从零学习 JSSP 派工规则并跨规模泛化       | 析取图、GIN、PPO、候选动作集合、增量奖励             | 支持图表示、合法候选集和 PPO 调度；其静态 makespan 奖励不能直接照搬 | 方法参考 |
| M05 | A Multi-Action Deep Reinforcement Learning Framework for Flexible Job-Shop Scheduling Problem                                               | Kun Lei; Peng Guo; Wenchao Zhao; Yi Wang; Linmao Qian; Xiangyin Meng; Liansheng Tang | 2022 | Expert Systems with Applications 205:117796                              | [DOI](https://doi.org/10.1016/j.eswa.2022.117796)                                                                                                                        | 同时决定工序和机器的 FJSP            | 多 MDP、MPGN、Multi-PPO、两个子策略          | 与“作业类型→具体飞机”的条件分层动作最接近                    | 方法参考 |
| M06 | Large-Scale Dynamic Scheduling for Flexible Job-Shop With Random Arrivals of New Jobs by Hierarchical Reinforcement Learning                | Kun Lei; Peng Guo; Yi Wang; Jian Zhang; Xiangyin Meng; Linmao Qian                   | 2024 | IEEE Transactions on Industrial Informatics 20(1):1007-1018（online 2023） | [DOI](https://doi.org/10.1109/TII.2023.3272661)                                                                                                                          | 随机新任务到达下的超大规模实时调度          | 高层分解动态子问题；低层 GNN 排序和 MLP 机器分配       | 支持按时间尺度/语义拆分决策，但扰动类型和资源结构与甲板不同            | 方法参考 |
| M07 | Smart Scheduling of Dynamic Job Shop Based on Discrete Event Simulation and Deep Reinforcement Learning                                     | Ziqing Wang; Wenzhu Liao                                                             | 2024 | Journal of Intelligent Manufacturing 35(6):2593-2610（online 2023）        | [DOI](https://doi.org/10.1007/s10845-023-02161-w)                                                                                                                        | 随机到达下的实时动态作业车间调度           | 离散事件仿真、PPO、事件决策、奖励缩放                | 是本项目“DES 环境+PPO 在线派工”最接近的方法参照之一           | 邻近问题 |
| M08 | Introducing PetriRL: An Innovative Framework for JSSP Resolution Integrating Petri Nets and Event-Based Reinforcement Learning              | Sofiene Lassoued; Andreas Schwung                                                    | 2024 | Journal of Manufacturing Systems 74:690-702                              | [DOI](https://doi.org/10.1016/j.jmsy.2024.04.028) / [arXiv:2402.00046](https://arxiv.org/abs/2402.00046)                                                                 | 用事件系统和 RL 联合处理 JSSP        | Petri 网、事件驱动 RL、Maskable PPO、动态合法动作 | 为“自动事件推进+决策点调用策略+动作掩码”提供直接实现依据            | 方法参考 |
| M09 | Offline Reinforcement Learning for Job-Shop Scheduling Problems                                                                             | Imanol Echeverria; Maialen Murua; Roberto Santana                                    | 2025 | Applied Soft Computing 184:113736                                        | [DOI](https://doi.org/10.1016/j.asoc.2025.113736) / [arXiv:2410.15714](https://arxiv.org/abs/2410.15714)                                                                 | 专家数据下纯 BC 泛化不足、在线 RL 样本效率低 | 异构图、可变动作、离线 actor-critic、奖励目标+BC 项  | 直接支持“模仿项不能替代优化目标”；与 BC 后 RL 微调思想相邻        | 方法参考 |
| M10 | Deep Reinforcement Learning for Solving Resource Constrained Project Scheduling Problems with Resource Disruptions                          | Hongxia Cai; Yunqi Bian; Lilan Liu                                                   | 2024 | Robotics and Computer-Integrated Manufacturing 85:102628                 | [DOI](https://doi.org/10.1016/j.rcim.2023.102628)                                                                                                                        | 可再生资源发生中断时的 RCPSP 动态决策     | MDP、GNN 状态编码、PPO                    | 与设备故障、人员容量下降和资源恢复后的事件驱动派工高度对应             | 邻近问题 |

## 4. 十篇重点论文精读

### 4.1 Ryan et al. (2014)：专家启发式与整数规划

- **问题**：在甲板计划需要频繁调整的场景中，比较资深操作员的经验规则与整数线性规划决策支持系统。
- **模型与算法**：操作者给定目标和约束，ILP 返回候选计划供批准；实验把该人机协同方案与专家启发式进行对照。
- **实验结论 \[文献结论]**：专家启发式经常优于优化器生成的计划，但也更保守。
- **局限 \[文献结论]**：研究关注计划质量和人机协同，不是随机闭环保障过程，也没有学习型策略。
- **迁移判断 \[本项目推断]**：Heuristic 不能只作为 BC 教师，还必须作为独立强基线；若 RL 只达到教师水平，不能据此宣称学习方法优越。

### 4.2 Su et al. (2018)：随机时长下的主动鲁棒调度

- **问题**：在作业持续时间随机、人员/装备/工作空间/补给资源受限时，同时追求较短工期与较高按期完成概率。
- **模型与算法**：用 operation-on-node-flow 网络表示多机作业优先关系；构建串行/并行调度生成方案、鲁棒人员和装备分配，以及执行期预约束策略；使用 HTLBO 优化。
- **实验结论 \[文献结论]**：论文报告 HTLBO 在确定性案例和随机项目调度对比中优于所选算法，资源分配方案和预约束策略提高了鲁棒性。
- **局限 \[本项目推断]**：核心输出是基线计划及 makespan 风险，不是持续事件反馈下的策略；其仿真案例不能证明对本项目分布和目标有效。
- **迁移判断 \[本项目推断]**：适合定义随机持续时间、资源类别和鲁棒性指标；不应把其“最小 makespan”直接写成“最大固定时域放飞架次”的等价结论。

### 4.3 Yuan et al. (2018)：周期与事件联合滚动时域

- **问题**：保障系统状态变化和扰动发生后，如何实时修正多资源、多项目计划。
- **模型与算法**：整数规划描述可再生资源（人员、站位、设备）和不可再生资源（油料、气体、武器、电力）；周期滚动用于持续跟踪，事件触发机制避免无必要重排；DPGA 求解窗口内问题。
- **实验结论 \[文献结论]**：论文案例表明该框架可在动态环境中响应扰动，并改善实时决策。
- **局限 \[本项目推断]**：每次事件仍调用群智能重优化，计算时延随规模增长；论文没有学习一个可在毫秒级直接推理的策略。
- **迁移判断 \[本项目推断]**：本项目的 DES 决策点可视为更细粒度的事件触发滚动决策，CP-SAT 则可实现同一思想下的滚动优化对照。

### 4.4 Liu et al. (2020)：出动全过程与空间轨迹联合

- **问题**：把甲板初始布置、滑行、准备、弹射、起飞和挡焰板冷却/恢复纳入统一出动计划。
- **模型与算法**：将过程抽象为六阶段混合柔性流水车间；用双层遗传算法处理计划，并用伪谱法和优先避碰策略规划多机协同轨迹。
- **实验结论 \[文献结论]**：论文在 8 架未知初始布置和 13 架固定布置案例上获得满足约束的出动方案。
- **局限 \[文献结论]**：实验规模有限，重点是出动与移动，不含本项目的多波次回收后再保障闭环。
- **迁移判断 \[本项目推断]**：当前项目把空间过程压缩为标量转运时间是明确简化；未来若引入显式路径冲突，状态和动作结构都会显著变化，旧模型不能默认复用。

### 4.5 Su et al. (2023)：飞机与资源转移

- **问题**：克服“忽略飞机和保障资源在站位间转移”的常见简化。
- **模型与算法**：建立带转移的资源受限多飞机调度模型；比较串行和并行调度框架、机位优先规则及资源分配规则，并用改进遗传编程搜索活动优先规则。
- **实验结论 \[文献结论]**：公开摘要报告并行框架在多数实验中优于串行框架，平均优势为 10.19%；机位规则对串行框架影响较小。
- **局限 \[本项目推断]**：主要研究离线构造式调度，未解决随机故障发生后的策略泛化；结果依赖其转移模型与案例。
- **迁移判断 \[本项目推断]**：它为本项目当前的“固定机位转运时间函数”提供了最直接的扩展依据，也提示资源移动本身应占用时间和可达性，而不是任意新增惩罚。

### 4.6 Chen et al. (2025)：人类经验引导强化学习

- **问题**：舰载机保障调度面临稀疏奖励、高维状态动作和动态约束，纯 RL 探索效率低，差质量专家经验又可能误导。
- **模型与算法**：建立动态 MDP；构造人类经验数据库，在交互时用模式匹配纠正危险/错误动作；通过混合奖励和自适应权重把人类意图融入 actor-critic。
- **实验结论 \[文献结论]**：论文在三个仿真场景中与所选方法比较，报告更好的性能，并测试了次优经验引导下的鲁棒性。
- **局限 \[本项目推断]**：经验注入是在线干预，不等同于离线行为克隆预训练；公开摘要不足以证明该方法在跨分布波次和故障组合下仍稳定。
- **迁移判断 \[本项目推断]**：本项目应把 BC 教师质量作为独立变量，至少比较优质示范、混合质量示范以及无示范 PPO，避免把“专家引导”当作天然正收益。

### 4.7 Li et al. (2026)：GNN 与甲板作业 DRL

- **问题**：缓解传统方法在解质量与在线计算速度之间的矛盾。
- **模型与算法**：将甲板作业调度构造成 MDP，以 GNN 编码组合结构，由 DRL 策略直接从状态生成计划；论文还比较探索策略和折扣因子。
- **实验结论 \[文献结论]**：摘要报告 softmax 探索与折扣因子 1.0 是较稳健配置；策略优于所选优先规则，小规模接近元启发式，大规模搜索能力更强，并把决策时间从数十分钟降到秒级。
- **局限 \[本项目推断]**：公开元数据未显示通用公开基准或代码；实验目标、状态与当前项目的固定时域架次目标不完全相同。
- **迁移判断 \[本项目推断]**：论文证明“GNN+DRL 用于甲板调度”已经存在。本项目的可辨识贡献应落在闭环波次、增量架次奖励、层次动作、掩码及 BC→PPO 的组合与消融。

### 4.8 Li and Womer (2015)：随机 RCPSP 的闭环 ADP

- **问题**：随机活动时长下，开环活动序列难以利用执行中已经观测到的信息。
- **模型与算法**：用 rollout 构造近似动态规划策略；用 CP 改进优先规则基策略，并组合前看与回看近似。
- **实验结论 \[文献结论]**：在基准实例上获得有竞争力的解，尤其适用于非对称工期分布。
- **局限 \[文献结论]**：近似策略仍受基策略和价值近似质量影响；大状态空间下精确闭环动态规划不可行。
- **迁移判断 \[本项目推断]**：SampledRandom、滚动 CP-SAT 和 RL 都可统一理解为不同强度的闭环近似；这有利于在论文中解释各求解器，而不是把它们写成互不相关的方法集合。

### 4.9 Lei et al. (2022)：多动作 Multi-PPO

- **问题**：FJSP 每一步同时涉及“选工序”和“分配机器”，平坦动作空间组合爆炸。
- **模型与算法**：以析取图表示状态，MPGN 提取特征，Multi-PPO 训练两个相互关联的子策略。
- **实验结论 \[文献结论]**：作者在随机和公开基准上报告优于人工规则的解质量，并展示了面向大规模工序实例的推理效率。
- **局限 \[本项目推断]**：主要是静态 makespan，两个动作分支的信用分配并不自动适用于随机多波次甲板环境。
- **迁移判断 \[本项目推断]**：本项目可把联合策略明确写为 `π(type, aircraft|s)=π_type(type|s)π_aircraft(aircraft|s,type)`，分别实施类型掩码和条件飞机掩码，并在 PPO 中重算一致的 masked log-prob。

### 4.10 Lassoued and Schwung (2024)：PetriRL

- **问题**：让 RL 只处理真正需要选择的调度事件，由形式化离散事件模型自动处理确定性状态转移。
- **模型与算法**：Petri 网表示作业与资源约束；自动变迁推进系统时间；关键资源分配点调用 Maskable PPO；允许推理时动态加入工序。
- **实验结论 \[文献结论]**：在公开与随机 JSSP 实例上表现出跨规模泛化和竞争性；消融专门检验了事件控制和动作掩码。
- **局限 \[本项目推断]**：基础 JSSP 的资源和状态机仍明显简单于甲板回收、并行保障和波次窗口。
- **迁移判断 \[本项目推断]**：这是当前 `advance_to_next_event()`、合法动作掩码和“无可执行动作才推进时间”设计最强的方法学支撑之一。

## 5. 按主题整理的研究现状

### 5.1 舰载机甲板调度

**\[文献结论]**

1. 早期工作已把出动生成抽象为 fork-join 排队网络，并把甲板重规划视为带时间约束的资源分配问题 \[D01-D02]。
2. 2016--2020 年形成了多种建模范式：HTN、扩展 FJSP、RCMPSP、混合柔性流水车间，以及资源配置与计划联合优化 \[D03-D05, D07-D09]；2023 年又出现直接面向出动率的 DES \[D06]。
3. 不确定性研究主要采用随机持续时间、区间时长、主动鲁棒计划、滚动时域和基线-反应式修复 \[D05, D07, D10；另见第 9 节]。
4. 近年模型开始显式处理飞机/资源转移、牵引路径、站位占用和起降协同 \[D11, D13, D15, D20]。
5. 2023 年以后已有舰载机专用 MARL、经验引导 RL、HRL、安全 RL、GNN+DRL 和端到端 DRL \[D12, D14, D16-D19]。

**\[本项目推断]**

- 直接文献并非空白，但样本集中、场景定义不统一，公开环境和跨论文可复现实验较少。
- 多数研究优化单批次 makespan、总等待或延误；本项目的“固定时域累计成功放飞架次”更接近吞吐量/长期产出目标。
- 本项目不应与包含完整甲板几何、滑行避碰和牵引动力学的模型混为一谈；当前更准确的定位是随机资源调度模型。

### 5.2 资源受限与随机调度

**\[文献结论]**

- 随机 RCPSP 的关键区分是开环基线计划和闭环策略。闭环 ADP 利用已实现工期更新决策 \[M01]。
- 舰载机文献中的主动鲁棒调度通过资源冗余、预约束或稳定性指标吸收扰动 \[D05；另见第 9 节 Su et al.]；滚动时域和基线-反应式方法在扰动后重算或局部修复 \[D07, D10]。
- 随机多项目 ADP 可以同时处理项目到达、随机工期、多类资源与延误收益 \[M03]。
- 资源中断 RCPSP 已有 GNN+PPO 方法，把扰动后的项目和资源状态纳入闭环决策 \[M10]。

**\[本项目推断]**

- A/B 波次可视作有结构的动态项目到达；每架飞机是带 precedence 的任务链，人员/设备/站位是可再生资源。
- 当前目标不是最小化一个项目 makespan，而是在有限时域中最大化完成的 launch 数量，因此不能直接采用 RCPSP 的单项目终止奖励。
- 故障与资源中断扩展应优先选择有物理意义的事件：飞机不可用、设备停机、人员容量暂降、回收失败/复飞；每项参数必须有场景依据或敏感性分析。

### 5.3 离散事件仿真优化

**\[文献结论]**

- 仿真优化把目标和约束视作由随机仿真器评价的黑箱；算法选择取决于决策是离散还是连续、仿真成本、输出数量和噪声结构 \[M02]。
- DES 与 PPO 可以在作业完成或到达等事件点连接，实现在线动态派工 \[M07]。
- PetriRL 表明，确定性事件推进和可行动作决策可以分离，动作掩码与事件控制都对学习效果有独立价值 \[M08]。
- 舰载机领域已有直接的出动生成 DES，以及 O2DES 与 DRL/PSO 两阶段耦合案例 \[D06, D15]。

**\[本项目推断]**

- 当前环境的事件队列不是单纯实验工具，而是 MDP 的时间推进机制。策略应只在合法决策点被调用，中间过程自动跳到下一事件。
- 对多个求解器的公平比较需要共同随机数或配对种子、多个独立重复、均值/标准差或置信区间。固定 `seed=10007` 可作为统一基准实例，但不能单独支撑随机优越性结论。
- 仿真预算较高时，可用 SampledRandom、代理模型或 ADP 做候选筛选，但最终性能必须回到同一个真实 DES 上评估。

### 5.4 强化学习与模仿学习调度

**\[文献结论]**

- 析取图/GNN/PPO 已被用于从状态直接构造 JSSP 调度，并能在训练规模之外泛化 \[M04]。
- 多动作和层次 RL 能把工序选择、机器分配或动态子问题划分拆开 \[M05-M06]。
- 舰载机领域已经出现多智能体、三层 HRL、人类经验引导、安全 RL 和端到端 DRL \[D12, D14, D16-D19]。
- 行为克隆在调度中容易受专家分布覆盖限制；离线 RL 通过同时保留优化目标与模仿项缓解纯 BC 的局限 \[M09]。
- 非法动作掩码已有专门的策略梯度理论和实验研究；PetriRL 也在事件调度中对其进行了消融 \[M08；另见第 9 节 Huang and Ontañón]。

**\[本项目推断]**

- 尚未发现高相关正式论文完整复现“启发式示范进行 BC 预训练，然后以 PPO 在线微调”的舰载机闭环调度流程。该组合可以作为方法贡献，但必须通过消融证明，而不能仅凭两个组件各自有文献依据。
- BC 数据不应只有教师在熟悉分布上的最优轨迹，还应覆盖扰动、次优状态和恢复动作；否则 PPO 微调可能只是在教师分布附近震荡。
- 分层策略的两层概率、掩码和熵应分别记录。否则无法判断性能瓶颈来自作业类型选择、飞机选择还是非法动作处理。

## 6. 研究空白

### 6.1 文献能够支持的空白

1. **目标函数不一致**：直接文献多数以 makespan、总等待、延误、鲁棒性或多目标综合指标为主；固定时域内最大化成功放飞架次的 RL 对比证据有限。
2. **过程覆盖碎片化**：出动、回收、维修、滑行和资源配置常被分别研究；公开论文中完整覆盖回收后并行加油/挂弹再放飞闭环的工作较少。
3. **动态扰动与学习脱节**：鲁棒/滚动调度文献处理随机工时和扰动，但多用元启发式；RL 文献常在较规则的仿真环境中验证，故障与资源中断覆盖不足。
4. **分层动作已有先例但语义不同**：现有 HRL 多按流程/站位/资源或子问题/工序/机器分层，尚不能直接证明“作业类型→飞机”是本场景最优分解。
5. **BC 与 PPO 组合证据不足**：调度领域存在 BC、离线 RL、PPO 和经验引导方法，但“BC 预热后 PPO 微调”尚未形成舰载机调度中的标准方案。
6. **复现与统计报告不足**：直接文献较少公开统一环境、实例生成器和代码，不同论文的场景与指标难以横向比较。

### 6.2 必须标为本项目推断的空白

以下表述是可检验的研究假设，不是现有文献已经证明的事实：

- **\[本项目推断]** 事件驱动决策比固定时间步决策更适合当前环境，因为大量时间推进不包含可选动作。
- **\[本项目推断]** 作业类型与飞机对象的条件分层策略能够降低平坦动作空间的学习难度。
- **\[本项目推断]** 启发式 BC 预训练能提高 PPO 初期样本效率，但能否超越教师取决于奖励归因、探索覆盖和 PPO 微调强度。
- **\[本项目推断]** 在 60 分钟波次间隔的高压区间，算法差异更容易被观测；这一判断只能由当前环境的敏感性实验验证，不能外推为一般规律。
- **\[本项目推断]** 奖励绑定“新增成功放飞架次”比等待时间、资源利用率等代理奖励更符合当前单一目标，并可能减少奖励归因偏移。

## 7. 毕业论文定位建议

### 7.1 推荐题目

**面向舰载机多波次保障闭环的事件驱动分层强化学习调度方法**

可选副标题：

**基于行为克隆预训练、动作掩码与 PPO 微调的离散事件仿真研究**

### 7.2 问题定位

将问题定义为：

> 有限时域、随机处理时间、共享可再生资源约束下的多波次舰载机闭环动态调度问题；飞机按“回收—停放—加油与挂弹并行—放飞”演化，策略仅在事件时点选择“作业类型—具体飞机”，目标是最大化期望成功放飞架次。

该定位与 FJSP/RCMPSP 相邻，但不能简单等同：

- 有 A/B 波次窗口和周期性回收/放飞；
- 一个飞机会跨波次反复进入保障链；
- 加油与挂弹形成 fork-join；
- 目标是有限时域吞吐量，而非单批工件 makespan；
- 可行动作集合随事件、资源和飞机状态动态变化。

### 7.3 可主张的贡献

以下贡献必须由实验完成后再写成“已证明”：

1. **问题建模**：给出面向多波次保障闭环的事件驱动 MDP，并保持优化问题与求解器分离。
2. **动作结构**：设计“作业类型→具体飞机”的条件分层策略，对每层实施状态相关动作掩码。
3. **训练流程**：用强启发式轨迹进行行为克隆预热，再以 PPO 围绕成功放飞增量进行在线优化。
4. **统一比较**：在同一 DES、同一随机种子集合和同一时间预算下比较 Random、FIFO、SPT、EDD、Heuristic、SampledRandom、CP-SAT 与 RL。
5. **扰动评估**：对随机工时、飞机故障、设备不可用和资源容量下降进行分级压力测试，报告性能与恢复能力。

不应主张：

- “首次将 RL 用于舰载机调度”；
- “首次提出舰载机分层强化学习”；
- “首次考虑随机性或动态重调度”；
- 在未实现路径和碰撞约束时声称“完整甲板仿真”。

### 7.4 建议研究问题

- RQ1：事件驱动决策是否比固定步长决策减少无效交互并提高样本效率？
- RQ2：分层动作与平坦动作相比，是否改善收敛、合法动作率和跨规模泛化？
- RQ3：BC 预训练是否只加速达到教师水平，还是能帮助 PPO 稳定超越教师？
- RQ4：增量放飞奖励是否比 makespan/等待时间代理奖励更稳定地优化最终架次？
- RQ5：在随机工时、设备故障和资源下降下，RL 相对启发式与滚动 CP-SAT 的优势是否保持？

### 7.5 最小实验矩阵

训练消融：

| 组别 |        BC | PPO | 分层动作 |     动作掩码 |
| -- | --------: | --: | ---: | -------: |
| A  |         否 |   是 |    是 |        是 |
| B  |         是 |   否 |    是 |        是 |
| C  |   是（仅初始化） |   是 |    是 |        是 |
| D  | 是（持续衰减正则） |   是 |    是 |        是 |
| E  |         否 |   是 |    否 |        是 |
| F  |         否 |   是 |    是 | 否或非法动作惩罚 |

统一指标：

- 成功放飞架次：主指标；
- 错失架次、等待时间、资源利用率：解释性指标；
- 非法动作率、每回合决策次数、策略熵：训练诊断；
- 多随机种子均值、标准差、置信区间和配对差值：统计指标；
- 推理时间、训练样本数和墙钟时间：效率指标。

统一基准：

- 保留项目规定的 `seed=10007` 和 60 分钟波次间隔作为固定高压锚点；
- 另外使用独立种子集合估计随机性能，避免单种子偶然性；
- 所有求解器共享完全相同的实例、事件序列生成规则、时域和资源容量。

## 8. 结论

**\[文献结论]** 舰载机甲板调度已经形成从排队网络、数学规划、项目/车间调度、鲁棒与滚动优化，到多智能体和深度强化学习的连续研究链。2025--2026 年出现的舰载机专用 HRL、安全 RL、GNN+DRL 和端到端 DRL，使“使用 RL”本身不再构成充分创新。

**\[本项目推断]** 本项目最稳妥的毕业论文定位是：在明确简化边界的资源调度模型中，研究固定时域架次目标下的事件驱动、分层掩码策略，以及 BC→PPO 相对强启发式、滚动 CP-SAT 和随机规划基线的增益与适用边界。论文价值应来自可复现的统一模型、严格消融、随机扰动评估和诚实的负结果，而不是扩大场景描述或弱化基线。

## 9. 引用追踪说明与边界文献

以下文献在引用链中有价值，但因篇幅、重复性或问题边界未进入 30 篇主表：

- Ryan et al., *Designing an Interactive Local and Global Decision Support System for Aircraft Carrier Deck Scheduling*, AIAA Infotech\@Aerospace 2011, [DOI](https://doi.org/10.2514/6.2011-1516)：D02 的早期会议版本/系统背景。
- Su et al., *A Robust Scheduling Optimization Method for Flight Deck Operations of Aircraft Carrier With Ternary Interval Durations*, IEEE Access 2018, [DOI](https://doi.org/10.1109/ACCESS.2018.2879503)：以三元区间替代精确概率分布，适合数据不足时的鲁棒建模。
- Cui et al., *A Dual Population Multi-Operator Genetic Algorithm for Flight Deck Operations Scheduling Problem*, JSEE 2021, [DOI](https://doi.org/10.23919/JSEE.2021.000028)：与 D09、D11 的元启发式路线重复度较高。
- Liu et al., *Optimization of Fixed Aviation Support Resource Station Configuration for Aircraft Carrier Based on Aircraft Dispatch Mission Scheduling*, CJA 2023, [DOI](https://doi.org/10.1016/j.cja.2022.06.023)：更偏资源站配置。
- Wang et al., *A Review on Carrier Aircraft Dispatch Path Planning and Control on Deck*, CJA 2020, [DOI](https://doi.org/10.1016/j.cja.2020.06.020)：适合完整甲板空间建模综述，但不是本项目当前资源调度核心。
- Lee and Kim, *Graph-Based Imitation Learning for Real-Time Job Shop Dispatcher*, IEEE TASE 2025, [DOI](https://doi.org/10.1109/TASE.2024.3486919)：是纯 BC 调度的重要补充，建议在扩展 BC 实验时精读。
- Luo et al., *Real-Time Scheduling for Dynamic Partial-No-Wait Multiobjective Flexible Job Shop by Deep Reinforcement Learning*, IEEE TASE 2022, [DOI](https://doi.org/10.1109/TASE.2021.3104716)：提供多目标层次 RL 参照，但本项目当前是单一目标。
- Lassoued et al., *Flexible Manufacturing Systems Intralogistics: Dynamic Optimization of AGVs and Tool Sharing Using Coloured-Timed Petri Nets and Actor-Critic RL with Actions Masking*, Journal of Manufacturing Systems 2025, [DOI](https://doi.org/10.1016/j.jmsy.2025.06.017)：共享运输资源和工具与甲板资源冲突同构，可用于后续资源移动扩展。
- Huang and Ontañón, *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, FLAIRS 2022, [DOI](https://doi.org/10.32473/flairs.v35i.130584)：证明标准掩码可产生合法策略梯度，并展示非法动作比例增大时掩码相对惩罚法的优势。
- Chakrabortty et al., *An Event-Based Reactive Scheduling Approach for the Resource Constrained Project Scheduling Problem with Unreliable Resources*, Computers & Industrial Engineering 2021, [DOI](https://doi.org/10.1016/j.cie.2020.106981)：可作为故障后非学习型事件重排基线。

中文补充文献：

- 刘敖、刘克，**《舰载机甲板作业调度研究进展》**，《系统工程理论与实践》2017, 37(1):49-60，[DOI](https://doi.org/10.12011/1000-6788%282017%2901-0049-12)：较早系统归纳国内外甲板调度的系统仿真、系统设计、系统优化与路径规划研究。
- 李亚飞、吴庆顺、徐明亮等，**《舰载机保障作业实时调度：一种强化学习方法》**，《中国科学：信息科学》2021, 51(2):247-262，[DOI](https://doi.org/10.1360/SSI-2020-0316)：舰载机保障实时强化学习的中文直接研究。
- 李亚飞、高磊、郝宏杰等，**《舰载机保障作业人机协同决策方法》**，《中国科学：信息科学》2023, 53(12):2493-2510，[DOI](https://doi.org/10.1360/SSI-2022-0403)：补充人类经验参与保障调度决策的国内研究线。
- 郭方、韩维、刘玉杰等，**《基于可变作业流程的舰载机维护保障调度》**，《航空学报》2025, 46(13):531195，[DOI](https://doi.org/10.7527/S1000-6893.2024.31195)：考虑人员、设备、作业流程及武器协同取送，采用改进粒子群优化求解。

## 10. 建模：以 Yoon 2023 为论文环境基线

### 10.1 基线选择

主环境基线选用：

> Hee Chang Yoon, Seung Heon Oh, Jong Hun Woo, Jung-Hoon Chung, Hyuk Lee, Sun-Ah Jung. *Discrete Event Simulation of Aircraft Sortie Generation on an Aircraft Carrier*. Winter Simulation Conference, 2023. [DOI](https://doi.org/10.1109/WSC60868.2023.10407756) / [公开全文](https://informs-sim.org/wsc23papers/204.pdf)

选择理由：

1. 研究对象直接是航母舰载机出动生成过程和出动率，而不是通用车间问题；
2. 采用离散事件仿真，环境模型与调度算法相互分离；
3. 将系统定义为固定数量飞机持续循环的封闭系统，与当前多波次循环结构一致；
4. 使用可配置的 SGP 作业流程、FlyPro 飞行任务计划和甲板位置图；
5. 把飞机、流程、位置、策略、数据和事件日志拆分，便于接入不同求解器；
6. 论文全文公开，模型结构和案例设计可以直接核验。

辅助参照不作为主基线：

- Yuan et al. (2018)：用于事件触发滚动时域、资源故障和动态重调度；
- Zhang et al. (2026)：用于多波次流程层—站位层—资源层 HRL 对照；
- Su et al. (2018)：用于随机工时和鲁棒性指标；
- Guo et al. (2026)：用于起飞与回收共享区域冲突。

### 10.2 对齐原则

“向论文环境靠拢”指对齐模型语义和可验证行为，不要求复制其软件实现。

- 保留当前 `heapq` 事件引擎，不因论文使用 SimPy 而重写仿真内核；
- 将论文环境和项目环境保存为两个独立配置，不覆盖现有可复现实验；
- 环境定义独立于 Random、Heuristic、CP-SAT 和 RL 等求解器；
- 论文未给出的参数不得猜成事实，应标记为复现实验假设；
- 所有新增约束必须有论文或前置需求依据，不为放大算法差距人为增加限制。

### 10.3 论文环境与当前环境映射

| Yoon 2023 环境要素 | 当前环境 | 建模动作 |
|---|---|---|
| 固定机队在 SGP 中循环 | 45 架共享飞机、A/B 波次需求循环 | 保留封闭机队语义 |
| 可配置的 pre-launch/post-launch SGP | 固定 `R/F/M/L` 状态机 | 将流程定义与状态转移参数化 |
| FlyPro：任务类型、编队规模、开始时间、持续时间、取消阈值 | 固定 A/B 交替波次和 missed sortie | 抽象统一的任务计划输入 |
| 甲板离散图 | 停机位编号和标量转运时间 | 增加节点、边、距离和占用关系 |
| 舰艏/舰艉停机区、通道、跑道、升降机 | 停机位、单放飞通道、单回收通道 | 增加位置类型和位置资源 |
| 飞机在节点之间移动 | 移动被合并到挂弹转运时间 | 增加移动开始/完成事件和路径占用 |
| 预定运行策略组件 | `choose_action()` 求解器接口 | 保留并作为论文策略与项目策略的统一接口 |
| SGR 与事件日志 | sorties、missed、wave records | 补充单位时间 SGR、流程时间和瓶颈统计 |
| 机库维修与升降机通行 | 未实现 | 放入后续扩展，不进入第一轮复现 |

### 10.4 分阶段计划

#### B0：论文基线环境复现

目标：建立不依赖 RL 的 Yoon 2023 结构化 DES 基线。

工作内容：

1. 从论文提取 SGP、FlyPro、甲板图、服务时间和案例参数；
2. 定义六类核心实体：`Aircraft`、`Process`、`Location`、`Mission`、`Resource`、`Event`；
3. 把作业流程改为配置驱动，环境根据流程定义生成合法后继作业；
4. 把甲板表示为离散图，移动时间由路径距离决定；
5. 实现论文中的预定运行策略，先不接入 RL；
6. 输出 SGR、任务取消数、流程时间、资源利用率、位置等待和事件日志；
7. 复现论文案例中“飞机数量/甲板布局变化影响 SGR”的方向性结果。

验收标准：

- 飞机数量守恒，一架飞机不能同时位于多个位置或执行多个互斥作业；
- 任意时刻资源和位置占用不超过容量；
- 同一输入和随机种子产生相同事件序列；
- 改变飞机数量或甲板布局后，SGR 和等待时间呈现与论文案例一致的变化方向；
- 论文参数不完整时，只要求复现趋势，不虚构精确数值一致性。

#### B1：接入本项目闭环调度问题

目标：在论文基线上恢复当前项目的研究问题，而不是把项目改成论文的算法复刻。

保留内容：

- A/B 多波次需求标签及每波动态 20 架任务分配；
- `回收 -> 停放 -> 加油 || 挂弹 -> 放飞` 闭环；
- 固定时域最大化成功放飞架次；
- `作业类型 -> 具体飞机` 分层动作；
- action masking 和事件点决策；
- Random、FIFO、SPT、EDD、Heuristic、SampledRandom、CP-SAT、RL 统一接口。

需要形成两个明确配置：

| 配置 | 用途 | 目标 |
|---|---|---|
| `paper_yoon_2023` | 论文环境复现 | 验证 DES 结构与论文案例趋势 |
| `project_core` | 本项目统一算法实验 | 最大化固定时域成功放飞架次 |

B1 验收标准：

- 两个配置共用同一事件引擎和资源守恒逻辑；
- 所有求解器均能在两个配置上运行或明确声明不适用原因；
- `project_core` 的历史基准结果可以复现；
- 论文配置和项目配置的差异由配置与文档显式列出，不写入求解器内部。

#### B2：向前置环境逐层扩展

扩展顺序：

1. 甲板离散图、路径占用和移动时间；
2. 4 个起飞位，以及位于着舰区的起飞位与回收跑道之间的资源冲突；
3. 设备暂时不可用、人员容量下降等资源中断；
4. 回收时限、复飞队列和空中燃油约束；
5. 75 架全机队、故障替换和机库补充；
6. 飞机故障、维修、飞机升降机和机库循环；
7. 海况、人员岗位技能、轮班与疲劳；
8. 弹药型号、库存、兼容性和装配区容量。

每一层均需：

- 单独实验分支；
- 明确物理依据与参数来源；
- 对所有求解器使用相同环境；
- 做参数敏感性与消融实验；
- 记录模型维度、旧模型兼容性和实验结论。

### 10.5 保持不变的研究口径

1. 当前基础模型仍是资源调度模型，不称为完整甲板仿真；
2. 唯一优化目标保持为固定时域内期望成功放飞架次最大化；
3. missed sorties、等待时间和资源利用率只作为解释指标；
4. `seed=10007`、60 分钟波次和 12 波次继续作为统一高压评估锚点；
5. 同时增加多随机种子实验，单个固定种子不用于证明随机优越性；
6. 环境扩展与算法改动分开评估，避免无法判断性能变化来源。

### 10.6 暂不执行的事项

- 不立即把飞机规模从甲板 45 架扩展到全机队 75 架；
- 不同时加入海况、故障、轮班、路径和弹药库存；
- 不为追求“更复杂”而引入无来源约束；
- 不要求第一阶段精确复现论文未公开的参数；
- 不在论文环境复现完成前重新比较 RL 优劣。

### 10.7 当前实施状态

已在独立分支 `experiment/yoon-sgp-baseline` 完成 B0.1 和 B0.2：

1. 新建 `proj/doc/yoon_2023_environment_spec.md`，记录论文实体、参数、8 个案例、缺失参数和验收标准；
2. 新建 `env/scenario.py`，定义处理时间分布、SGP 作业、任务计划、位置节点、带权边和场景配置；
3. 将当前标量停机位转运时间映射为等价位置图，保持 `project_core` 历史行为；
4. 新增事件日志、每小时 SGR 和 sortie completion rate；
5. 新增 `scripts/inspect_scenario.py`，用于检查论文配置而不启动求解器；
6. 新增独立的 `YoonSortieGenerationEnv`，实现固定翼/SAR 任务、取消事件、
   节点级位置占用、逐节点移动、跑道优先级、机库/升降机和计划维修；
7. 新增 `scripts/run_yoon_baseline.py`，支持八案例、多随机种子、置信区间
   和事件日志导出。

验证结果：

- 项目 Conda 环境下 22 项单元/集成测试全部通过；
- `seed=10007`、60 分钟、12 波次基准与主分支一致：Random 141、FIFO 143、SPT 140、EDD 143、Heuristic 152、SampledRandom 144、CP-SAT 140；
- 当前代码变化只增加场景结构、图距离等价表示、事件日志和统计指标，没有改变 `project_core` 的调度结果。
- 结构复现已完成八案例各 100 次实验；16 架的大布局案例接近论文值，
  但 20 架案例明显偏低，尚未复现论文的相对趋势。

下一步 B0.3 将针对论文未公开的甲板边权、移动速度、完整 FlyPro 和运行
策略做参数辨识与敏感性分析。当前结果只能称为带显式假设的结构复现，
不能作为论文数值结果的精确复现。

### 10.8 B1：动态共享机队

已在 `experiment/dynamic-shared-fleet` 分支完成第一轮共享机队改造：

1. `project_core` 从 40 架永久 A/B 分组改为 45 架共享甲板机队；
2. A/B 保留为交替波次需求标签，每波需求仍为 20 架；
3. 任意已完成油弹保障的飞机都可填充当前波次任务席位；
4. 最后 5 架标记为初始备用角色，但不受额外放飞限制；
5. 下一波回收集合由上一波实际放飞飞机 ID 决定；
6. missed sortie 改为未填满的匿名任务席位；
7. RL 观测由永久组别改为备用角色、波次填充率和待回收压力，旧 checkpoint
   不再兼容。

在 `seed=10007`、60 分钟、12 波次条件下，Random/FIFO/SPT/EDD/Heuristic/
SampledRandom/CP-SAT 分别完成 143/152/157/158/157/143/162 架次。该结果
说明 5 架共享缓冲扩展了可行调度空间，但没有使所有策略都达到满波次。

### 10.9 B2.1：甲板空间图约束

已在 `experiment/spatial-deck-graph` 分支完成第一版空间扩展：

1. 将 45 个停机位划分为起飞区 10 个、着舰区 10 个和保障区 25 个；
2. 增加 4 个起飞位、1 个着舰跑道节点和 19 个通道节点；
3. 所有位置节点容量为 1，路径或目的节点冲突时由 action mask 屏蔽动作；
4. 放飞增加“停机位 → 起飞位”的滑行事件，回收增加“跑道 → 停机位”
   的转运时间；
5. Heuristic、SPT 和 CP-SAT 使用含路径时间的动作时长；
6. RL 全局观测增加空闲通道比例，观测 schema 升级为 3。

其中，停机位分区、4 个起飞位和 1 条着舰跑道来自前置需求；19 个通道
节点借鉴 Yoon 2023 的结构规模。具体邻接关系、每条边 1 分钟及“移动时
整条路径原子预留”均是结构复现实验假设，不是已知真实甲板参数。

固定 `seed=10007`、60 分钟、12 波次时，开启空间图后的
Random/FIFO/SPT/EDD/Heuristic/SampledRandom/CP-SAT 完成架次为
66/74/90/53/93/77/86；关闭空间图的严格匹配对照为
143/152/157/158/157/143/162。吞吐量下降幅度说明当前原子路径预留较为
保守，不能据此推断真实航母出动能力。下一步需要优先获取或校准甲板邻接、
边移动时间和逐节点移动规则，再决定是否将该约束并入主模型。
