# 2026-09-28 数据基线

本目录保存可提交的小型控制文件，不保存完整行情：

- `calendar_day.txt`：冻结交易日历；
- `instruments_all.txt`：冻结 Qlib 全量列表；bootstrap 会筛出 5181 只模型股票；
- `security_master.csv`：证券元数据快照；
- `MANIFEST.json`：正式数据目录的内容摘要。

从 BaoStock 重新拉取数据后执行：

```bash
quantx-reward data-manifest verify --data-root /path/to/data
```

只有校验通过时，才可声称输入数据与 2026-09-28 正式基线逐文件一致。
BaoStock 若事后修订前复权历史，校验会失败，此时仍可运行策略，但结果属于新数据版本。
