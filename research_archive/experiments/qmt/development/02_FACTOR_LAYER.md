# 因子层开发计划

> **版本**: v0.1.0
> **日期**: 2026-06-25
> **预计工期**: 3-5 天
> **依赖**: 数据层 (01_DATA_LAYER.md)

---

## 1. 目标

因子层负责因子定义、计算、存储和复用：
1. 封装 Qlib 表达式引擎，支持字符串表达式定义因子
2. 因子注册中心，管理因子元信息
3. 复杂因子（表达式无法表达）的 Python 实现 + DuckDB 存储
4. 因子缓存管理，避免重复计算

---

## 2. 文件清单与功能

```
quantx/core/factor/
├── __init__.py              # 模块入口
├── base.py                  # 因子抽象基类 + 元信息
├── registry.py              # 因子注册中心（单例）
├── dependency.py            # 因子依赖图管理（拓扑排序、变更影响分析）
├── expression.py            # Qlib 表达式引擎封装（ExpressionFactor 继承 Factor）
├── engine.py                # 惰性因子计算引擎（核心）
├── duckdb_store.py          # DuckDB 复杂因子存储
├── operators.py             # 自定义算子库（扩展 Qlib 表达式）
├── alpha101.py              # Alpha101 因子（Python 实现）
├── technical.py             # 技术指标因子（Python 实现）
└── validator.py             # 因子值校验
```

### 2.2 各文件详细说明

---

#### 2.2.1 `base.py` — 因子抽象基类

**功能**：定义因子的统一接口和元信息结构。

**核心类**：`FactorMeta`, `Factor`

```python
@dataclass
class FactorMeta:
    name: str                    # 因子名，如 'alpha005'
    category: str                # momentum / value / volatility / quality / technical
    depends_on: List[str]        # 依赖的原始字段或因子名
    description: str             # 中文描述
    version: str = '1.0.0'
    tags: List[str] = field(default_factory=list)

class Factor(ABC):
    meta: FactorMeta

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值，输入 MultiIndex (date, symbol) DataFrame，返回 MultiIndex Series"""
        ...

    def validate(self, result: pd.Series) -> Tuple[bool, str]:
        """校验因子值：覆盖率 > 80%，无异常值"""
        coverage = result.notna().mean()
        if coverage < 0.8:
            return False, f'Coverage {coverage:.1%} < 80%'
        if (np.abs(result) > 1e10).any():
            return False, 'Contains extreme values'
        return True, 'OK'
```

**开发要点**：
- 直接复用 `myquant-clean/dataset/alpha101.py` 的 Factor 基类设计
- `depends_on` 可以是原始字段名（如 `close`）或其他因子名（如 `factor:alpha005`）
- 因子值统一返回 `(date, symbol)` MultiIndex Series

---

#### 2.2.2 `registry.py` — 因子注册中心

**功能**：管理所有已注册的因子，支持按名称、分类查询。

**核心类**：`FactorRegistry`（单例）

```python
class FactorRegistry:
    _factors: Dict[str, Type[Factor]] = {}

    @classmethod
    def register(cls, factor_cls: Type[Factor]): ...
    @classmethod
    def get(cls, name: str) -> Type[Factor]: ...
    @classmethod
    def list_by_category(cls, category: str) -> List[str]: ...
    @classmethod
    def list_all(cls) -> List[FactorMeta]: ...
    @classmethod
    def auto_discover(cls, package: str): ...
```

**注册方式**：装饰器注册（推荐）
```python
@register_factor(
    name='alpha005', category='momentum',
    depends_on=['close', 'volume'],
    description='(-1 * ts_rank(rank(close), 10))'
)
class Alpha005(Factor):
    def compute(self, data): ...
```

**开发要点**：
- 直接复用 `myquant-clean/straregys/configs.json` 的注册思路 + alpha-gpt 的元信息管理
- 注册时自动校验：因子名唯一性、元信息完整性、依赖字段存在性
- 支持自动发现：扫描 `quantx.core.factor` 包下所有 Factor 子类

---

#### 2.2.2b `dependency.py` — 因子依赖图管理

**功能**：管理因子间的依赖关系，确保计算顺序正确。

**核心类**：`FactorDependencyGraph`

```python
class FactorDependencyGraph:
    def __init__(self, registry: FactorRegistry):
        self.graph = {}  # factor_name -> [dependent_factors]
        self._build(registry)

    def _build(self, registry):
        for name, factor in registry._factors.items():
            for dep in factor.meta.depends_on:
                if dep not in self.graph:
                    self.graph[dep] = []
                self.graph[dep].append(name)

    def get_compute_order(self, factor_names: List[str]) -> List[str]:
        """拓扑排序：返回正确的计算顺序，确保依赖先计算"""
        # 1. 收集所有需要计算的因子（含间接依赖）
        all_needed = set()
        for name in factor_names:
            self._collect_deps(name, all_needed)
        # 2. 拓扑排序
        return self._topological_sort(list(all_needed))

    def get_affected_factors(self, changed_factor: str) -> List[str]:
        """变更影响分析：当 X 因子变更时，返回所有受影响的下游因子"""
        affected = set()
        stack = [changed_factor]
        while stack:
            f = stack.pop()
            for downstream in self.graph.get(f, []):
                if downstream not in affected:
                    affected.add(downstream)
                    stack.append(downstream)
        return list(affected)

    def detect_cycle(self) -> Optional[List[str]]:
        """循环依赖检测：返回第一个检测到的循环路径"""
        # 使用 DFS 检测环
        ...

    def _collect_deps(self, name, collected):
        """递归收集所有依赖"""
        if name in collected:
            return
        collected.add(name)
        factor = self.registry.get(name)
        for dep in factor.meta.depends_on:
            if dep in self.registry._factors:  # 依赖其他因子
                self._collect_deps(dep, collected)
```

**开发要点**：
- 约 80 行新代码
- 依赖解析规则：
  - 依赖原始字段（如 `close`）：不触发额外计算，因为原始字段总是可用
  - 依赖其他因子（如 `factor:alpha005`）：递归收集，确保依赖因子先计算
  - 依赖市场级因子（如 `factor:market_breadth`）：同样递归收集
- 循环依赖检测在注册时自动执行，发现循环直接报错
- 变更影响分析在因子重新计算时使用——只重算受影响的下游因子

---

#### 2.2.3 `expression.py` — Qlib 表达式引擎封装

**功能**：封装 Qlib 的表达式引擎，让用户用字符串定义因子。`ExpressionFactor` 继承 `Factor` ABC，统一通过 `FactorRegistry` 管理。

**核心类**：`ExpressionFactor(Factor)`

```python
class ExpressionFactor(Factor):
    """基于 Qlib 表达式引擎的因子，继承 Factor ABC"""

    def __init__(self, meta: FactorMeta, expression: str):
        self.meta = meta
        self.expression = expression

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """通过 Qlib 的 D.features() 计算表达式"""
        # 从 data 的 MultiIndex 中提取 symbols 和日期范围
        symbols = data.index.get_level_values('symbol').unique().tolist()
        start = str(data.index.get_level_values('date').min().date())
        end = str(data.index.get_level_values('date').max().date())
        return D.features(symbols, [self.expression], start, end, freq='day')

# 使用示例：一行字符串定义一个因子
ret_5d = ExpressionFactor(
    FactorMeta(name='ret_5d', category='momentum', depends_on=['close'],
               description='5日收益率'),
    expression='Ref($close, -5) / $close - 1'
)
```

**开发要点**：
- 约 60 行新代码
- `ExpressionFactor` 继承 `Factor` ABC，与 Python 因子走同一个注册体系
- 表达式因子自动享受 Qlib 的 `DiskExpressionCache` 缓存机制
- 常用 Qlib 算子：`Ref`, `Mean`, `Std`, `Sum`, `Max`, `Min`, `Rank`, `Scale`, `Delta`, `Corr`, `Cov`

---

#### 2.2.4 `duckdb_store.py` — DuckDB 复杂因子存储

**功能**：为表达式引擎无法表达的复杂因子（Python 实现）提供持久化存储。

**核心类**：`DuckDBFactorStore`

```python
class DuckDBFactorStore:
    """DuckDB 因子值存储，长表格式 (date, stock, factor_name, value)"""

    def __init__(self, db_path: str = 'data/warehouse/factors.db'):
        self.conn = duckdb.connect(db_path)
        self._init_tables()

    def store(self, factor_name: str, data: pd.Series): ...
    def load(self, factor_names: List[str], start, end, stocks=None) -> pd.DataFrame:
        """返回宽表 MultiIndex DataFrame"""
    def has_coverage(self, factor_name, start, end) -> bool: ...
    def get_missing_ranges(self, factor_name, start, end) -> List[Tuple[str, str]]: ...
```
**开发要点**：
- 长表格式：`(date DATE, stock_code VARCHAR, factor_name VARCHAR, value DOUBLE)`
- 主键：`(date, stock_code, factor_name)`
- 索引：`(factor_name, date)`
- 查询时按需 pivot 为宽表

---

#### 2.2.5 `engine.py` — 惰性因子计算引擎

**功能**：管理因子计算、缓存和加载，确保回测时所有需要的因子都可用。

**核心类**：`LazyFactorEngine`

```python
class LazyFactorEngine:
    def __init__(self, registry: FactorRegistry, store: DuckDBFactorStore):
        self.registry = registry
        self.store = store
        self.dep_graph = FactorDependencyGraph(registry)

    def ensure_factors(
        self, factor_names: List[str], start: str, end: str, stocks: List[str]
    ) -> pd.DataFrame:
        """确保所有因子已计算并缓存，返回宽表 MultiIndex DataFrame"""
        # 1. 解析依赖图：获取所有需要计算的因子（含间接依赖）
        all_needed = self.dep_graph.get_compute_order(factor_names)

        # 2. 按类型分组
        expr_factors = []  # 表达式因子
        py_factors = []    # Python 因子
        for name in all_needed:
            factor = self.registry.get(name)
            if isinstance(factor, ExpressionFactor):
                expr_factors.append(name)
            else:
                py_factors.append(name)

        results = {}

        # 3. 表达式因子：批量通过 Qlib D.features() 计算
        #    Qlib 自动缓存，无需手动管理
        if expr_factors:
            expr_strings = [self.registry.get(n).expression for n in expr_factors]
            expr_result = D.features(stocks, expr_strings, start, end, freq='day')
            for i, name in enumerate(expr_factors):
                results[name] = expr_result.iloc[:, i]

        # 4. Python 因子：按拓扑顺序逐个计算
        for name in py_factors:
            factor = self.registry.get(name)

            # 4a. 检查 DuckDB 缓存
            coverage = self.store.has_coverage(name, start, end)

            if not coverage:
                # 4b. 计算缺失的日期范围
                missing = self.store.get_missing_ranges(name, start, end)
                for m_start, m_end in missing:
                    # 4c. 收集依赖数据（已计算的结果 + 原始数据）
                    dep_data = self._collect_dep_data(factor, results, m_start, m_end, stocks)
                    # 4d. 计算因子值
                    computed = factor.compute(dep_data)
                    # 4e. 存入 DuckDB
                    self.store.store(name, computed)

            # 4f. 从 DuckDB 加载完整范围
            results[name] = self.store.load([name], start, end, stocks)

        # 5. 只返回请求的因子（不含中间依赖），合并为宽表
        return pd.concat([results[name] for name in factor_names if name in results], axis=1)

    def _collect_dep_data(self, factor, results, start, end, stocks):
        """收集因子计算所需的依赖数据"""
        dep_data = {}
        for dep in factor.meta.depends_on:
            if dep in results:
                # 依赖其他因子：使用已计算的结果
                dep_data[dep] = results[dep].loc[start:end]
            else:
                # 依赖原始字段：从 Qlib 加载
                dep_data[dep] = D.features(stocks, [f'${dep}'], start, end, freq='day')
        return pd.DataFrame(dep_data)
```

**开发要点**：
- 约 150 行新代码
- 区分表达式因子和 Python 因子，走不同的计算路径
- 表达式因子无需缓存管理（Qlib 自动处理）
- Python 因子按需计算 + DuckDB 存储 + 增量更新
- 因子版本管理：当 `FactorMeta.version` 变更时，自动失效缓存
  ```python
  def store(self, name, data):
      meta = self.registry.get(name).meta
      self.conn.execute("""
          INSERT OR REPLACE INTO factor_meta (name, version, updated_at)
          VALUES (?, ?, NOW())
      """, [name, meta.version])
      # ... 存储因子值
  ```

---

## 3. 开发顺序

```
Day 1: base.py + registry.py + dependency.py（因子框架 + 注册 + 依赖管理）
Day 2: expression.py（Qlib 表达式封装）+ duckdb_store.py（DuckDB 存储）
Day 3: engine.py（惰性计算引擎）+ alpha101.py（Alpha101 因子移植）
Day 4-5: technical.py（技术指标）+ validator.py（因子校验）+ 测试
```

## 4. 验收标准

- [ ] 表达式因子一行字符串定义，无需写 Python 代码
- [ ] Python 因子计算结果自动缓存到 DuckDB，二次查询不重复计算
- [ ] 因子注册中心支持按分类查询
- [ ] 因子值覆盖率 > 80%
- [ ] 内置 Alpha101 因子（至少 20 个）
