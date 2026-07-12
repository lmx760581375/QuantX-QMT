# QuantX

A股量化研究与回测系统。

## 架构

- **数据层**: BaoStock → Qlib 二进制格式，零外部依赖
- **因子层**: 表达式引擎 + DuckDB 复杂因子存储
- **回测引擎**: Policy 架构（选股/调仓/执行），规则+模型双模式
- **策略层**: 三个 Policy 接口（StockSelector / RebalanceStrategy / ExecutionStrategy）

## 快速开始

```bash
# 安装依赖
pip install -e .

# 配置远程 xqshare（本地 .env 不会提交到 Git）
cp .env.example .env

# 同步数据
python -m quantx.tools.sync_data --start 2020-01-01

# 运行回测
python -m quantx.tools.run_backtest --strategy strategies/ma_cross.py --start 2020-01-01 --end 2025-12-31
```

QMT 数据通过 Windows 上运行的 xqshare 服务获取。`.env` 使用 xqshare 原生配置项：

```dotenv
XQSHARE_REMOTE_HOST=192.168.0.117
XQSHARE_REMOTE_PORT=18812
XQSHARE_CLIENT_ID=client-standard
XQSHARE_CLIENT_SECRET=replace-with-your-client-secret
```

可以通过 `QUANTX_ENV_FILE=/path/to/custom.env` 为不同环境指定其他配置文件。

## 开发状态

- [x] 数据层
- [ ] 因子层
- [ ] 回测引擎
- [ ] 策略层
- [ ] 评估层
- [ ] 工具链

## 许可

MIT
