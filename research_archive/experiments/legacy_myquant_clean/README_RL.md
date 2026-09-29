# 强化学习多因子权重优化系统

基于PPO算法的量化交易策略，使用强化学习动态调整多因子权重，实现最优的股票组合配置。

## 系统架构

```
myquant-clean/
├── config.yaml              # 配置文件（所有参数集中管理）
├── train_rl.py             # 训练脚本
├── test_rl.py              # 测试/评估脚本
├── run_training.sh         # 一键运行脚本
├── requirements.txt        # 依赖包列表
├── rl_model/              # RL核心模块
│   ├── __init__.py
│   ├── ppo_agent.py       # PPO算法实现
│   ├── backtest_env.py    # 回测环境封装
│   └── utils.py           # 工具函数
└── straregys/
    └── strategy_rl_weight.py  # RL权重策略
```

## 核心特性

### 🎯 动态因子权重优化
- 使用PPO算法学习最优因子权重分配
- 基于市场状态实时调整权重
- 支持多因子组合优化

### 📊 智能状态表示
- 历史市场数据特征提取
- 因子统计特征（均值、标准差、偏度、峰度等）
- 市场整体指标（收益率、波动率等）

### 💰 组合奖励函数
- 总收益率（年化）
- 最大回撤（风险控制）
- 夏普比率（风险调整收益）

### 🔧 可配置化设计
- 所有参数通过YAML配置文件管理
- 支持不同市场环境和因子配置
- 灵活的网络架构和超参数设置

## 快速开始

### 1. 环境准备

```bash
# 创建并激活conda环境
conda create -n rl_quant python=3.8
conda activate rl_quant

# 安装依赖
pip install -r requirements.txt

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. 配置参数

编辑 `config.yaml` 文件，设置训练参数：

```yaml
experiment:
  name: "factor_weight_rl"
  seed: 42

data:
  factor_names: ["alpha036", "alpha072"]  # 要优化的因子
  stock_pool: ["000001.SZ", "000002.SZ", ...]  # 股票池
  train_start_date: "20200101"
  train_end_date: "20221231"
  test_start_date: "20230101"
  test_end_date: "20231016"

ppo:
  learning_rate: 0.0003
  gamma: 0.99
  clip_epsilon: 0.2

training:
  num_episodes: 100
  episode_length: 252  # 一年交易日
```

### 3. 运行训练

```bash
# 一键运行训练和测试
chmod +x run_training.sh
./run_training.sh

# 或者分别运行
# 训练
python train_rl.py --config config.yaml --experiment_name my_experiment

# 测试
python test_rl.py --model_path outputs/rl_experiments/best_model.pth --config config.yaml
```

## 使用说明

### 训练阶段

```bash
python train_rl.py [参数]

参数选项:
  --config CONFIG           配置文件路径 (默认: config.yaml)
  --experiment_name NAME    实验名称
  --resume PATH            恢复训练的模型路径
  --seed SEED              随机种子
  --log_level LEVEL        日志级别 (DEBUG/INFO/WARNING/ERROR)
```

### 测试阶段

```bash
python test_rl.py [参数]

参数选项:
  --model_path PATH        训练好的模型路径
  --config CONFIG          配置文件路径
  --start_date DATE        测试开始日期
  --end_date DATE          测试结束日期
  --output_dir DIR         输出目录
  --plot                   生成性能图表
```

## 配置详解

### 状态空间配置

```yaml
environment:
  lookback_days: 30        # 历史回顾天数
  state_features:
    factor_stats: true     # 因子统计特征
    market_stats: true     # 市场统计特征
    use_market_index: false # 是否使用指数数据
```

状态向量包含：
- 因子统计特征：每个因子的均值、标准差、偏度、峰度、分位数（共6个特征 × 因子数量）
- 市场统计特征：收益率均值、波动率、成交量均值、波动率（4个特征）

### 动作空间配置

使用Dirichlet分布建模权重，确保权重和为1：

```yaml
ppo:
  network:
    hidden_dims: [256, 128]  # 网络隐藏层
    activation: "relu"       # 激活函数
```

### 奖励函数配置

```yaml
environment:
  reward_weights:
    return_weight: 10.0     # 收益率权重
    drawdown_weight: -5.0   # 回撤权重（负数）
    sharpe_weight: 1.0      # 夏普率权重
```

奖励计算公式：
```
reward = return_weight × total_return - drawdown_weight × max_drawdown + sharpe_weight × sharpe_ratio
```

## 输出结果

### 训练输出
- `outputs/rl_experiments/{experiment_name}_{timestamp}/`
  - `training.log` - 训练日志
  - `training_stats.csv` - 训练统计
  - `best_model.pth` - 最佳模型
  - `final_model.pth` - 最终模型
  - `config.yaml` - 训练配置备份

### 测试输出
- `outputs/test_results/test_{timestamp}/`
  - `performance.csv` - 绩效指标
  - `daily_balance.csv` - 每日账户余额
  - `trades.csv` - 交易记录
  - `weights_history.csv` - 权重变化历史
  - `performance_plot.png` - 性能图表（可选）

## 性能指标

### 回测指标
- **总收益率**：策略的总收益百分比
- **年化收益率**：年化收益百分比
- **最大回撤**：历史最大亏损比例
- **夏普比率**：风险调整收益指标
- **交易次数**：总交易笔数
- **胜率**：盈利交易占比

### RL训练指标
- **平均奖励**：每个回合的平均奖励
- **策略损失**：Actor网络损失
- **价值损失**：Critic网络损失
- **熵损失**：动作分布熵（探索度量）

## 扩展开发

### 添加新因子
1. 在因子数据文件中添加新因子数据
2. 在 `config.yaml` 中更新 `factor_names` 列表
3. 系统自动适应新的因子数量

### 修改状态特征
在 `backtest_env.py` 的 `_extract_state_features` 方法中添加新的特征提取逻辑。

### 自定义奖励函数
在 `backtest_env.py` 的 `_calculate_final_reward` 方法中修改奖励计算逻辑。

### 网络架构调整
在 `config.yaml` 中修改 `ppo.network` 配置，或在 `ppo_agent.py` 中自定义网络结构。

## 注意事项

### 数据要求
- 确保因子数据文件存在且格式正确
- 股票池中的股票要有完整的历史数据
- 训练和测试期间数据要连续且无缺失

### 性能优化
- GPU加速：确保CUDA可用，模型会自动使用GPU
- 内存优化：适当调整 `lookback_days` 避免状态空间过大
- 并行化：考虑使用多进程进行并行回测

### 风险控制
- 设置合理的止盈止损规则
- 监控最大回撤和夏普比率
- 定期重新训练模型适应市场变化

## 故障排除

### 常见问题

1. **内存不足**
   - 减少 `lookback_days`
   - 缩小股票池
   - 使用更小的网络架构

2. **训练不稳定**
   - 调整学习率
   - 增加价值损失系数
   - 使用梯度裁剪

3. **收敛慢**
   - 增加训练回合数
   - 调整奖励权重
   - 优化状态特征

### 日志分析
- 查看 `training.log` 了解训练过程
- 监控损失函数变化
- 分析奖励函数分布

## 引用

基于以下论文和实现：
- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- 金融强化学习: [Deep Reinforcement Learning for Automated Stock Trading](https://arxiv.org/abs/2005.12559)

---

如有问题，请查看日志文件或提交Issue。
