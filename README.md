HRL-GAT基于图注意力网络与强化学习的影响力最大化框架（Hybrid Reinforcement Learning with GAT for Influence Maximization）。

在 WIC（Weighted Independent Cascade）扩散模型下，通过 GAT 编码图结构、ECMR 筛选候选种子集、PPO 智能体进行序贯种子选取。

项目结构

```
HRL-GAT/
├── data/                  # 图数据集（如 Email.csv）
├── models/
│   ├── gat.py             # GAT 编码器（对比+平滑预训练）
│   └── policy.py          # Actor-Critic 网络
├── rl/
│   ├── agent.py           # PPO 智能体
│   └── env.py             # WIC 扩散环境 & ECMR 筛选
├── utils/
│   ├── config.py          # 超参数配置
│   ├── data_loader.py     # 数据加载与特征提取
│   └── misc.py            # 工具函数
├── train.py               # 训练入口
├── test.py                # 测试入口（Monte Carlo 评估）
└── requirements.txt
```

训练（k = 5, 10, ..., 50）
python train.py

测试
python test.py


数据格式

在 `data/` 下放置边表 CSV 文件，两列分别为源节点和目标节点 ID：


source,target
0,1
0,2
1,3


