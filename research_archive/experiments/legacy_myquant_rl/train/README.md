# Quant RL Dataset

这个模块实现了量化交易强化学习的数据集处理功能。

## 功能特性

1. **数据获取**: 通过xtquant接口获取股票复权数据
2. **数据预处理**: 计算相对涨跌幅等特征
3. **数据集分割**: 支持训练集和测试集的时间窗口分割
4. **PyTorch Dataset**: 兼容PyTorch数据加载器，支持滑动窗口采样

## 数据格式

### 原始数据格式
```python
{stock_code: pd.DataFrame}
```
DataFrame包含以下字段：
- `time`: 时间戳
- `open/high/low/close`: 价格数据
- `volume/amount`: 成交量数据
- `adj_*`: 复权后的价格数据
- `turnover`: 换手率

### 处理后数据格式
```python
{stock_code: pd.DataFrame}
```
DataFrame包含以下字段：
- `date`: 日期
- `stock_code`: 股票代码
- `rel_adj_open/high/low/close/vwap`: 相对前一天的涨跌幅
- `rel_volume`: 相对前一天的成交量涨跌幅
- `turnover`: 换手率

### Dataset输出格式
```python
{
    'history_data': torch.Tensor,  # (window_size, stock_num, feature_num)
    'today_data': torch.Tensor,    # (stock_num, feature_num)
    'tomorrow_data': torch.Tensor, # (stock_num,) - 明天收益率，用于loss计算
    'stock_ids': torch.Tensor,     # (stock_num,)
    'date': str,                   # 当前日期
    'stock_list': List[str]        # 股票列表
}
```

## 使用方法

### 基本使用

```python
from train.dataset import QuantDataLoader

# 创建数据加载器
data_loader = QuantDataLoader(cache_dir='./data_cache', force_refresh=False)

# 定义训练和测试时间窗口
train_spans = [
    ('20200101', '20201231'),
    ('20210101', '20211231'),
]
test_spans = [
    ('20220101', '20221231'),
]

# 创建数据集
train_dataset, test_dataset = data_loader.create_datasets(
    train_spans=train_spans,
    test_spans=test_spans,
    window_size=30,  # 历史30天数据
    stock_list=None,  # 使用全部股票，设为['000001.SZ', '000002.SZ']使用指定股票
    start_time='20150101',
    end_time='20221231'
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 方法1: 通过索引获取样本（兼容PyTorch DataLoader）
sample = train_dataset[0]  # 仍然支持索引访问
print(f"历史数据形状: {sample['history_data'].shape}")  # (30, stock_num, 7)
print(f"当天数据形状: {sample['today_data'].shape}")    # (stock_num, 7)

# 方法2: 通过日期直接获取样本
valid_dates = train_dataset.get_valid_dates()
if valid_dates:
    sample = train_dataset[valid_dates[0]]  # 通过日期访问
    print(f"通过日期获取: {sample['date']}")

# 方法3: 获取整个时间窗口的数据（推荐用于batch处理）
window_data = train_dataset.get_time_window_data('20210201', '20210205')
print(f"时间窗口数据样本数: {len(window_data)}")
```

### 与PyTorch DataLoader结合使用

```python
from torch.utils.data import DataLoader

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 使用DataLoader
for batch in train_loader:
    history_data = batch['history_data']  # (batch_size, window_size, stock_num, feature_num)
    today_data = batch['today_data']      # (batch_size, stock_num, feature_num)
    stock_ids = batch['stock_ids']        # (batch_size, stock_num)

    # 传递给模型...
```

### 高级用法：按时间窗口获取数据

```python
# 获取指定时间窗口的所有数据
window_data = train_dataset.get_time_window_data('20200101', '20201231')

# 这将返回整个时间窗口的所有样本，每个样本包含一天的数据
for sample in window_data:
    history_data = sample['history_data']  # (window_size, stock_num, feature_num)
    today_data = sample['today_data']      # (stock_num, feature_num)
    date = sample['date']                  # 日期字符串

    # 处理单个日期的数据...
```

## 配置参数

- `cache_dir`: 缓存目录，用于存储处理后的数据
- `force_refresh`: 是否强制重新下载和处理数据
- `window_size`: 历史数据窗口大小（天数）
- `stock_list`: 指定股票列表，None表示使用全部股票
- `start_time/end_time`: 数据时间范围

## 数据缓存

模块会自动缓存原始数据和处理后的数据到指定目录：
- `raw_data.pkl`: 原始数据缓存
- `processed_data.pkl`: 处理后数据缓存

设置`force_refresh=True`可以强制重新下载数据。

## 依赖环境

使用anaconda3/envs/test环境：
```bash
${HOME}/anaconda3/envs/test/bin/python
```

## 注意事项

1. 确保数据目录`${HOME}/git_projects/myquant-main/data`存在且包含股票数据
2. 首次运行时会下载和处理数据，可能需要一些时间
3. 数据量大时建议使用缓存功能
