# 日线数据合同

## 数据源

正式日线源为 BaoStock：

- frequency：日线；
- adjustflag：`2`，前复权；
- 起始日期：2010-01-01；
- 冻结截止日期：2026-09-28；
- 股票列表：`artifacts/reference/data_20260928/instruments_all.txt` 中供 Reward
  特征仓使用的 5181 只股票，排除指数。

`security_master.csv` 随数据目录生成，供名称和元数据扩展使用；当前 Reward 特征与正式回测
不依赖其中的行业或名称字段，板块类型由证券代码规则确定。

数据源可能事后修订历史前复权价格。因此“命令执行成功”不等于“严格复现”；
必须同时通过冻结数据 manifest。

## 原始 CSV

布局：

```text
data/raw/baostock/stocks/<SYMBOL>.csv
```

必要字段：

- `date`
- `open/high/low/close/preclose`
- `volume/amount`
- `turnover`
- `tradestatus`
- `pct_chg`
- `is_st`

## Qlib provider

布局：

```text
data/qlib_data_fixed/
├── calendars/day.txt
├── instruments/all.txt
└── features/<symbol>/
    ├── open.day.bin
    ├── high.day.bin
    ├── low.day.bin
    ├── close.day.bin
    ├── volume.day.bin
    ├── vwap.day.bin
    ├── factor.day.bin
    └── change.day.bin
```

其中：

```text
vwap = amount / (volume * 100)
change = (close - preclose) / preclose
factor = 1.0
```

前复权已经由 BaoStock 完成；Qlib 转换不得再次复权。

## 数据版本校验

`artifacts/reference/data_20260928/MANIFEST.json` 固定记录：

- 日历 SHA-256；
- A 股 instruments 合同 SHA-256；
- 5181 只 Reward 特征股票原始 CSV 聚合 SHA-256；
- 对应 Qlib feature 聚合 SHA-256；
- 文件数量与总字节数。

执行：

```bash
quantx-reward data-manifest verify --data-root /path/to/data
```

任何一项不一致，都不能把后续回测称为对 2026-09-28 基线的严格复现。
最终合并 score 还会单独校验 SHA-256：

```text
ea0d6ee77fd42ab4e555b1ed6256db010af38b0b8aba13a6928ea315ffa80534
```

## 接口

全量初始化：

```bash
quantx-reward bootstrap-data \
  --data-root /path/to/data \
  --start 2010-01-01 \
  --end 2026-09-28
```

复权变化感知的增量更新：

```bash
quantx-reward update-data \
  --data-root /path/to/data \
  --end YYYY-MM-DD
```

增量更新会重新获取重叠窗口。如果历史前复权价格改变，则从该股票最早本地日期重新拉取，
并重写该股票的 Qlib bin，避免旧复权坐标与新数据拼接。
