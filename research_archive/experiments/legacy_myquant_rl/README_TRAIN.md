# 量化交易强化学习训练脚本

## 概述

这是一个基于强化学习的量化交易训练脚本，使用FinancialFactorModel来预测股票得分，通过最大化未来收益来训练模型。

## 核心特性

1. **监督学习框架**: 使用未来N天的累积收益率作为训练目标
2. **选股策略**: 选择logits softmax最大的top_k只股票
3. **多指标优化**: 同时考虑收益率和风险指标（夏普率、最大回撤）
4. **完整训练流程**: 包含数据加载、训练、验证、模型保存

## 训练原理

### 数据流
```
历史数据(30天) + 当天数据 → 模型 → 股票得分logits → softmax → 选择top_k股票 → 计算未来收益率
```

### 损失函数
```python
# 主要损失：最大化选中股票的未来收益率
return_loss = -portfolio_return.mean()

# 辅助损失：多样性和熵正则化
total_loss = return_weight * return_loss + diversity_loss + entropy_loss
```

### 选股策略
```python
# 对logits进行softmax
probs = torch.softmax(logits, dim=-1)

# 选择概率最大的top_k只股票
_, top_indices = torch.topk(probs, top_k, dim=-1)
```

## 文件结构

```
train/
├── train_script.py      # 主要训练脚本
├── dataset.py          # 数据集定义
├── stock_tokenizer.py  # 股票编码器
└── ...

model/
└── models.py           # 模型定义

data_cache/             # 缓存目录
├── raw_data_*.pkl      # 原始数据缓存
├── processed_data_*.pkl # 处理后数据缓存
└── stock_tokenizer.json # 股票编码映射

checkpoints/            # 模型保存目录
├── best_model_*.pth    # 最佳模型
└── final_model.pth     # 最终模型
```

## 使用方法

### 1. 基本训练

```python
from train.train_script import main
main()
```

### 2. 自定义配置

```python
from train.train_script import QuantTrainer, create_data_loaders
from train.stock_tokenizer import build_stock_tokenizer

# 创建tokenizer
tokenizer = build_stock_tokenizer(cache_dir='./data_cache')

# 模型配置
model_config = {
    'state_dim': 7,
    'hidden_dim': 256,
    'stock_num': len(tokenizer),
    # ... 其他配置
}

# 训练配置
train_config = {
    'batch_size': 8,
    'num_epochs': 100,
    'future_days': 5,
    'top_k': 20,
    # ... 其他配置
}

# 创建训练器和数据加载器
trainer = QuantTrainer(model_config, train_config, tokenizer)
train_dataloader, val_dataloader = create_data_loaders(train_config)

# 开始训练
trainer.train(train_dataloader, val_dataloader)
```

## 配置参数

### 模型配置 (model_config)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| state_dim | 股票状态特征维度 | 7 |
| hidden_dim | 隐藏层维度 | 256 |
| stock_num | 股票数量 | 动态获取 |
| num_attention_heads | 注意力头数量 | 8 |
| global_state_num_layers | 全局状态层数 | 4 |
| global_state_top_k | 全局状态top_k | 50 |
| action_num_layers | Actor层数 | 2 |

### 训练配置 (train_config)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| batch_size | 批次大小 | 4 |
| num_epochs | 训练轮数 | 50 |
| learning_rate | 学习率 | 1e-4 |
| future_days | 预测未来天数 | 5 |
| top_k | 选择股票数量 | 20 |
| window_size | 历史窗口大小 | 30 |
| return_weight | 收益率权重 | 1.0 |
| sharpe_weight | 夏普率权重 | 0.1 |
| drawdown_weight | 回撤权重 | 0.05 |

## 数据格式

### 输入数据
```python
{
    'history_data': torch.Tensor,  # (batch_size, window_size, stock_num, feature_num)
    'today_data': torch.Tensor,    # (batch_size, stock_num, feature_num)
    'tomorrow_return': torch.Tensor, # (batch_size, stock_num)
    'date': List[str],             # 日期列表
    'stock_list': List[str],       # 股票代码列表
    # 预计算的未来指标
    'future_1d_return': torch.Tensor,      # 未来1天收益率
    'future_1d_max_drawdown': torch.Tensor, # 未来1天最大回撤
    'future_1d_sharpe': torch.Tensor,       # 未来1天夏普率
    'future_3d_return': torch.Tensor,      # 未来3天...
    'future_3d_max_drawdown': torch.Tensor,
    'future_3d_sharpe': torch.Tensor,
    'future_5d_return': torch.Tensor,      # 未来5天...
    'future_5d_max_drawdown': torch.Tensor,
    'future_5d_sharpe': torch.Tensor,
    'future_10d_return': torch.Tensor,     # 未来10天...
    'future_10d_max_drawdown': torch.Tensor,
    'future_10d_sharpe': torch.Tensor,
    'future_20d_return': torch.Tensor,     # 未来20天...
    'future_20d_max_drawdown': torch.Tensor,
    'future_20d_sharpe': torch.Tensor,
}
```

### 模型输出
```python
logits: torch.Tensor  # (batch_size, stock_num) - 每只股票的得分
```

## 预计算指标的优势

相比于训练时动态计算，预计算未来指标有以下优势：

1. **效率提升**: 数据预处理阶段完成计算，避免训练时的重复计算
2. **一致性**: 所有样本使用相同的计算方法，确保指标一致性
3. **灵活性**: 可以预计算多个时间周期的指标（1天、3天、5天、10天、20天）
4. **完整性**: 包含收益率、最大回撤、夏普率等多维度风险指标
5. **可扩展**: 容易添加新的技术指标或风险度量

### 预计算指标说明

- **future_Nd_return**: 未来N天的累积收益率
- **future_Nd_max_drawdown**: 未来N天内的最大回撤
- **future_Nd_sharpe**: 未来N天的夏普比率（简化为收益率标准差的倒数）

## 训练指标

训练过程中会监控以下指标：

- **train_loss**: 训练损失（包含收益率、回撤、夏普率等多目标）
- **portfolio_return**: 投资组合收益率
- **sharpe_ratio**: 夏普比率
- **max_drawdown**: 最大回撤
- **win_rate**: 胜率（收益率>0的比例）

## 测试

运行测试脚本验证功能：

```bash
python test_train.py
```

## 注意事项

1. **数据依赖**: 需要预先下载股票数据到`data_cache`目录
2. **内存使用**: 大规模股票池需要足够的GPU内存
3. **训练时间**: 完整训练可能需要数小时到数天
4. **超参数调优**: 需要根据具体情况调整模型和训练参数

## 扩展

### 添加新的指标
在`compute_portfolio_metrics`函数中添加新的风险指标。

### 修改选股策略
修改`select_stocks_by_logits`函数实现不同的选股逻辑。

### 自定义损失函数
在`compute_loss`函数中添加新的损失项。
