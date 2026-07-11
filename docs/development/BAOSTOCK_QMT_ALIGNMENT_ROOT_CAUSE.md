# BaoStock/QMT 收益差异根因

时间：2026-07-09

## 结论

这次 63 只股票样本里看到的 BaoStock 与 QMT 收益差异，根因不是两边日线价格本身相差很大，而是早期构造的 BaoStock partial provider 缺少策略 `look_back_days` 需要的前置历史。

坏 provider `data/qlib_data_baostock_partial63` 的交易日历从 `2020-01-02` 才开始，但 2020 冠军策略设置了 `look_back_days: 320`。当前引擎把这个参数解释为自然日回看窗口，因此实际加载起点是 `2019-02-16`；坏 provider 仍缺少整段回测起点之前的 warmup 数据。2016 版本更明显，实际加载起点是 `2015-02-18`，而坏 provider 仍从 `2020-01-02` 开始。

补足 2010 起始历史后，BaoStock 与 QMT fill 的收益已经基本对齐。

## 证据

审计脚本：

```powershell
C:\ProgramData\Anaconda3\envs\test\python.exe tools\audit_provider_lookback.py
```

输出文件：

```text
runs/baostock_qmt_alignment_audit.json
```

关键审计结果：

| 配置 | provider start | required load start | calendar gap | 缺失交易日 | coverage |
|---|---:|---:|---:|---:|---|
| `2020_champion_baostock_partial63.yaml` | `2020-01-02` | `2019-02-16` | `320` | `216` | fail |
| `2020_champion_baostock_partial63_2010.yaml` | `2010-01-04` | `2019-02-16` | `0` | `0` | pass |
| `2020_champion_qmt_fill_partial63.yaml` | `2010-01-04` | `2019-02-16` | `0` | `0` | pass |
| `2016_scale12_baostock_partial63.yaml` | `2020-01-02` | `2015-02-18` | `1779` | `1187` | fail |
| `2016_scale12_baostock_partial63_2010.yaml` | `2010-01-04` | `2015-02-18` | `0` | `0` | pass |
| `2016_scale12_qmt_fill_partial63.yaml` | `2010-01-04` | `2015-02-18` | `0` | `0` | pass |

收益对比：

| run | final value | total return | 说明 |
|---|---:|---:|---|
| `align63_2020_baostock` | `1.7598M` | `+75.98%` | 坏 BaoStock provider，缺 warmup |
| `align63_2020_baostock_2010` | `1.4765M` | `+47.65%` | BaoStock 补足 2010 历史 |
| `align63_2020_qmt_fill` | `1.4812M` | `+48.12%` | QMT fill，同样有足够历史 |
| `align63_2016_baostock` | `1.7533M` | `+75.33%` | 坏 BaoStock provider，实际从 2020 才跑 |
| `align63_2016_baostock_2010` | `1.3610M` | `+36.10%` | BaoStock 补足 2010 历史 |
| `align63_2016_qmt_fill` | `1.3573M` | `+35.73%` | QMT fill，同样有足够历史 |

因此，之前所谓 BaoStock 与 QMT 收益大幅不一致，是由不公平的 provider 覆盖范围造成的。

## 非根因项

已排除：

- 官方 qlib reader 与项目内 bin reader 差异：同一份 QMT fill provider 跑出的 `qmt_fill_mainboard_2020_champion` 与 `qmt_fill_mainboard_2020_champion_qlib_k1` 结果完全一致。
- QMT/BaoStock 单点前复权价格大幅不一致：抽样 `SZ002315`、`SH603399`、`SZ002568`、`SH601677`、`SZ002283`、`SH603009`，BaoStock `adjustflag=2` 与 QMT `front_ratio` 基本一致。

## 全量 BaoStock 状态

全量 BaoStock 同步已完成，`3194/3194` 成功、`failed=0`，并已转换到 provider：

```text
data/qlib_data_baostock_compare_full
```

状态文件：

```text
runs/baostock_compare_full_status.json
runs/baostock_compare_full_stderr.log
```

该全量 provider 的覆盖校验通过：2020 配置 `required_load_start=2019-02-16`，首个可用交易日为 `2019-02-18`，属于合法的非交易日 gap。

全量 BaoStock 2020 冠军回测曾启动一次，但超过 15 分钟工具超时，未落盘完整 run。基于当前审计和 63 只样本复跑，已能解释本次 BaoStock 收益差异的主要来源；全量回测可作为后续耗时复核单独跑。
