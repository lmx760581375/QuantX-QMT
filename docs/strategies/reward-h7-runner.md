# RM v1 Top5 H7 Runner

配置：`configs/strategies/reward_h7_runner.yaml`

## 选股与执行

- 全 A 股票池；
- `reward_score_7d` 每日降序 Top5；
- T 日 score，T+1 收盘执行；
- 组合目标权重模式，每个核心槽位约 20%；
- `max_position_weight=0.22` 只限制新增买入。

## 退出规则

1. 收益达到 10% 时卖出 85%；
2. 剩余 15% 作为 runner；
3. runner 从峰值回撤 4% 时卖出；
4. 最晚 H7 卖出。

部分止盈仓位不占核心 Top5 槽位，但仍占用真实现金和市值，也不会被重复买入。

截至 2026-06-02，该策略总收益 `+633.91%`、最大回撤 `-46.11%`、Sharpe `0.930`。
