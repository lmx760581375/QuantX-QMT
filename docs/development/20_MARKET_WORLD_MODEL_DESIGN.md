# Market World Model 设计方案

> 版本: v0.1.0
> 日期: 2026-07-12
> 状态: 设计提案
> 目标: 构建一个类似 VLA / World Model 的多模态市场表征系统，以历史 K 线图像、全局市场状态和股票行业概念为条件，通过 Pixel-space Diffusion 一次性生成未来 30 个交易日的累计收益路径，并支持多个下游 Decoder 共同增强 Encoder 表征。

---

## 1. 项目定义

本系统不是逐 token 生成下一根 K 线的传统 K 线 GPT，也不是在图像 tokenizer 的 latent 空间中生成未来图像。系统采用 Encoder-Decoder 结构：

```text
历史个股 K 线图像
  -> 通用 Image Tokenizer
  -> K-line Visual Tokens

全局市场信息
  -> Market Encoder
  -> Market Semantic / Quantitative Tokens

股票身份、行业和概念
  -> Asset / Industry / Concept Tokens

全部条件
  -> Multimodal Fusion Encoder
  -> Context Tokens

Context Tokens
  -> Pixel-space Diffusion Decoder
  -> 未来 30 个交易日累计收益路径图
```

系统与 VLA / World Model 的对应关系如下：

| World Model 概念 | 本系统中的含义 |
|---|---|
| Observation | 历史 K 线图像和历史市场状态 |
| Environment state | Market Encoder 输出的全局市场 tokens |
| Object / entity condition | 股票身份、行业和概念 tokens |
| Query | 对指定股票未来 30 个交易日的路径预测请求 |
| Future observation | 未来累计收益路径图 |
| World decoder | 条件式 Pixel-space Diffusion |

当前阶段不建模真实交易 action。未来若加入持仓、买卖动作、冲击成本和组合净值，系统才成为严格意义上的 action-conditioned financial world model。

## 2. 目标与非目标

### 2.1 核心目标

1. 学习跨股票、跨行业和跨市场状态可迁移的历史 K 线视觉表征。
2. 将高维全市场信息压缩成固定数量且具有预测价值的 Market Tokens。
3. 通过市场信息到文本的 VLM 式训练，为 Market Tokens 建立可解释语义。
4. 使用 Pixel-space Diffusion 一次性生成未来 30 日累计收益路径，避免逐日自回归 rollout 的误差传播。
5. 通过多任务 Decoder 约束共享 Encoder，使其不仅适用于路径生成，也适用于收益、风险、市场状态和横截面任务。
6. 输出一组可能的未来路径，而不是只输出单一点预测。

### 2.2 非目标

第一阶段不包含：

1. 逐日自回归生成 K 线。
2. 在 image tokenizer latent 空间执行未来路径 diffusion。
3. 直接生成未来绝对收盘价。
4. 将所有市场模态强制渲染为图片。
5. 使用未来 30 日信息动态调整输出画布坐标。
6. 直接将生成结果接入实盘交易。
7. 第一版同时引入新闻、公告、研报、分钟 K 线和订单簿等全部模态。

## 3. 已确定的设计约束

以下内容作为当前设计的固定前提，实施阶段不再作为默认分支反复讨论：

1. 历史 K 线图像使用通用 image tokenizer 编码。
2. Diffusion Decoder 在 pixel space 中直接生成未来路径图，不在 tokenizer latent space 中生成。
3. 主预测目标是未来 30 个交易日的累计对数收益路径。
4. 模型采用 Encoder-Decoder，而非逐 K 线自回归 GPT。
5. Market Encoder 独立于 K 线 Image Tokenizer，负责压缩全局市场状态。
6. 股票身份、行业和概念作为条件 tokens 参与融合。
7. 共享 Encoder 后续允许连接多个任务 Decoder。

仍需通过实验决定的内容统一记录在第 18 节，避免把尚未验证的选项写成实现事实。

## 4. 预测任务与因果边界

### 4.1 样本定义

对股票 `s` 和观测截止交易日 `t`，单个训练样本定义为：

```text
X_stock(s, <=t): 股票 s 截至 t 日可见的历史 K 线图像
X_market(<=t):   截至 t 日可见的全局市场信息
X_meta(s, t):    在 t 日有效的股票、行业和概念信息
Y(s, t+1:t+30): 未来 30 个交易日累计收益路径图
```

### 4.2 累计收益路径

未来第 `k` 个交易日的累计对数收益为：

```text
y_k = log(C_{t+k} / C_t),  k = 1, 2, ..., 30
```

模型生成的目标路径为：

```text
Y = [y_1, y_2, ..., y_30]
```

需要展示或回测价格时，再使用观测截止日收盘价恢复：

```text
C_hat_{t+k} = C_t * exp(y_hat_k)
```

累计收益路径比绝对 Close 更适合跨资产训练，因为它消除了股票价格尺度差异，并自然表达趋势、回撤和终点收益。

### 4.3 信息可见性

所有输入必须在 `t` 日样本截止时刻真实可得。数据管线需要明确样本是在收盘后预测，还是在下一交易日开盘前预测。第一版建议使用：

```text
样本时点: t 日收盘后
可见信息: t 日及以前的收盘行情、当日市场截面、当时有效的元数据
预测区间: t+1 至 t+30
```

以下情况必须防止：

1. 使用当前行业或概念标签回填历史样本。
2. 使用全量数据统计量归一化历史数据。
3. 使用未来路径最高点和最低点确定输出画布纵轴。
4. 将后来退市的股票从历史股票池中剔除。
5. 使用在 `t` 日收盘后尚未公开的资金流、财务或宏观数据。
6. 将复权因子变更错误地引入历史可见信息。

## 5. 总体架构

```text
                         +---------------------------+
历史 K 线图像 ---------->| 通用 Image Tokenizer      |
                         +-------------+-------------+
                                       |
                              K-line Visual Tokens
                                       |
                                       v
                         +-------------+-------------+
全局市场信息 ----------->| Market Encoder             |
                         | - Daily Cross-section      |
                         | - Temporal Market History  |
                         | - Semantic Queries         |
                         | - Quantitative Queries     |
                         +-------------+-------------+
                                       |
                              Market Context Tokens
                                       |
股票 / 行业 / 概念 --------------------+
                                       |
                                       v
                         +-------------+-------------+
                         | Multimodal Fusion Encoder  |
                         +-------------+-------------+
                                       |
                                 Context Tokens
                    +------------------+------------------+
                    |                  |                  |
                    v                  v                  v
            Pixel Diffusion      Return Decoder     Risk Decoder
                    |
                    v
          未来 30 日累计收益路径图
```

高内聚边界如下：

| 模块 | 单一职责 |
|---|---|
| Image Tokenizer | 将历史 K 线图像转换为通用视觉 tokens |
| Market Encoder | 将全局市场截面和历史压缩为固定数量的市场 tokens |
| Meta Encoder | 编码股票身份、层级行业和概念集合 |
| Fusion Encoder | 建立个股、市场和元数据之间的条件关系 |
| Pixel Diffusion | 在固定画布直接生成未来路径图 |
| Auxiliary Decoders | 从共享表征完成收益、风险和市场状态任务 |

## 6. 历史 K 线图像输入

### 6.1 输入目的

历史 K 线图像用于表达：

1. 趋势和趋势切换。
2. 局部高低点结构。
3. 跳空、实体和影线组合。
4. 波动率收缩或扩张。
5. 量价配合。
6. 不同时间尺度下的形态关系。

### 6.2 图像生成约束

虽然使用通用 image tokenizer，但输入图片必须由确定性渲染器生成，避免无意义的视觉变化：

```text
固定画布尺寸
固定 K 线数量
固定颜色和线宽
固定边距
固定成交量区域比例
无标题、坐标文字、水印和交互控件
缺失交易日和停牌使用稳定表示
```

第一版建议以日 K 为主，例如过去 256 个交易日渲染为一张图。是否增加周 K、月 K 或分钟 K，必须通过后续消融决定。

### 6.3 价格坐标

历史 K 线图可以使用窗口内相对价格坐标，但图像之外必须保留尺度条件，例如：

```text
历史波动率
ATR
窗口累计收益
窗口最大回撤
最新成交量尺度
```

这些尺度信息可由 Market/Meta 条件之外的 Stock Numeric Tokens 提供。否则形态相似但实际振幅完全不同的历史窗口可能被视觉编码器视为相同状态。

### 6.4 通用 Image Tokenizer

Image Tokenizer 的责任仅是把输入图像转成视觉 tokens：

```text
K-line Image
  -> Generic Image Tokenizer
  -> Token Grid / Token Sequence
  -> Projector
  -> K-line Visual Tokens
```

需要保留以下配置：

```text
tokenizer_name
tokenizer_version
image_size
patch_or_latent_grid_size
codebook_or_embedding_dim
frozen
normalization
```

第一阶段默认冻结 tokenizer，只训练 projector、Fusion Encoder 和下游任务。后续再通过冻结、仅微调 projector、解冻 tokenizer 后若干层三组实验判断领域适配收益。

## 7. Market Information Schema

Market Encoder 的输入不是一段自由文本，而是带明确时间戳的结构化市场状态。建议将其分为五组。

### 7.1 指数与基准序列

```text
上证指数
深证成指
沪深 300
中证 500
中证 1000
创业板指
科创 50
可选股指期货基差
```

每个指数至少包含收益、振幅、波动率、成交额变化和趋势位置。

### 7.2 市场宽度

```text
上涨 / 下跌 / 平盘家数
创新高 / 创新低家数
涨停 / 跌停家数
连续涨停梯队
不同收益分位区间的股票数量
全市场收益分布分位数
全市场成交额和换手率
横截面离散度
平均相关性或相关性代理指标
```

### 7.3 行业状态

对每个历史有效行业记录：

```text
行业收益
行业相对市场收益
行业成交额和换手率变化
行业上涨比例
行业波动率
行业内部离散度
行业动量和反转状态
行业估值或基本面聚合，可选
```

### 7.4 风格和风险状态

```text
大盘 / 小盘
成长 / 价值
动量 / 反转
高波动 / 低波动
高质量 / 低质量
流动性
市场 beta
拥挤度代理
```

### 7.5 股票横截面

Market Encoder 可以读取全市场股票状态，但不应把数千只股票直接平铺成无限长序列。单只股票的横截面特征可包括：

```text
1 / 5 / 20 日收益
历史波动率
成交额变化
换手率
相对行业收益
量价相关性
距离近期高低点
涨跌停和停牌状态
市值和风格暴露
```

股票横截面先按行业或其他稳定分组聚合，再交给 Market Encoder 的高层压缩模块。

## 8. Market Encoder

### 8.1 设计目标

Market Encoder 需要同时保留：

1. 可用自然语言描述的宏观市场语义。
2. 难以语言化但对未来个股路径有预测价值的连续统计信息。
3. 当日横截面的结构。
4. 一段历史时间内市场状态的形成、延续和切换。

因此不能只以文本生成作为唯一目标，也不能只用一个简单 pooling 向量承担全部信息。

### 8.2 分层压缩

推荐采用三阶段压缩：

```text
阶段 1: Daily Cross-sectional Encoder
  股票状态集合
    -> 行业内 Set Attention / Pooling
    -> Industry Tokens
    -> Daily Market Tokens

阶段 2: Temporal Market Encoder
  最近 N 日 Daily Market Tokens
    -> Temporal Transformer
    -> Historical Market Tokens

阶段 3: Query Compression
  Historical Market Tokens
    -> Q-Former / Perceiver Resampler
    -> 固定数量 Market Context Tokens
```

该结构支持每天增量计算并缓存 Daily Market Tokens，避免训练时反复编码完整股票截面。

### 8.3 Daily Cross-sectional Encoder

单日市场截面属于无序集合，编码器不应依赖股票输入顺序。推荐结构：

```text
Stock State Tokens
  -> Set Transformer / Attention Pooling
  -> Industry State Tokens

Industry State Tokens
+ Index Tokens
+ Breadth Tokens
+ Style Tokens
  -> Daily Market Transformer
  -> Daily Market Tokens
```

行业聚合不应只有均值。至少需要保留均值、分位数、离散度、尾部和样本数量，防止少量极端股票被平均值掩盖。

### 8.4 Temporal Market Encoder

Temporal Market Encoder 对最近一段历史的 Daily Market Tokens 建模：

```text
[MarketTokens_{t-N+1}, ..., MarketTokens_t]
  -> time/date embeddings
  -> Temporal Transformer
  -> Historical Market Tokens
```

时间 embedding 至少表达：

```text
相对交易日位置
绝对日期或市场阶段
周内位置
月末 / 季末等可选日历状态
```

绝对日期 embedding 需要谨慎，避免模型简单记忆训练阶段；可优先使用相对位置和周期性日历特征。

### 8.5 双类型输出 Tokens

Market Encoder 使用两组可学习 query：

```text
Semantic Queries
  -> 8～16 个 Semantic Market Tokens
  -> 对齐 LLM 和市场描述

Quantitative Queries
  -> 16～32 个 Quantitative Market Tokens
  -> 服务路径预测和数值辅助任务
```

两组 token 共享底层 Market Backbone，但具有不同监督目标。这样可以避免文本语义成为过强瓶颈，将语言难以表达的弱预测信号全部丢弃。

### 8.6 压缩率

Market Encoder 的目标是固定输出长度，而不是预先追求极限 token 数。建议从以下配置开始：

```text
Semantic Market Tokens: 16
Quantitative Market Tokens: 32
Total Market Tokens: 48
```

随后对 `64 / 48 / 32 / 16` 个总 token 做压缩率消融，比较语义质量、预测质量、训练吞吐和显存占用。

## 9. VLM 式市场语义对齐

### 9.1 训练路径

```text
Market Information
  -> Market Encoder
  -> Semantic Market Tokens
  -> LLM Projector / Q-Former
  -> LLM
  -> Structured Market State + Natural-language Summary
```

该任务的作用不是让 LLM 替代预测模型，而是为压缩后的市场 tokens 建立稳定、可解释的语义空间。

### 9.2 结构化市场描述

第一阶段优先自动生成可验证的结构化标签：

```text
市场方向
市场宽度
成交状态
主导风格
领涨行业
领跌行业
波动状态
相关性状态
行业分化程度
风险等级
```

示例：

```json
{
  "market_direction": "up",
  "market_breadth": "strong",
  "turnover_state": "contracting",
  "dominant_style": ["small_cap", "growth"],
  "leading_industries": ["electronics", "communication"],
  "lagging_industries": ["bank", "coal"],
  "volatility_state": "medium_high",
  "dispersion_state": "high"
}
```

然后由模板或受约束 LLM 转成自然语言摘要。结构化标签必须能从同一时点的市场数据确定性计算，避免人工文本风格差异和 LLM 幻觉污染监督。

### 9.3 训练目标

Market Encoder 的预训练损失建议为：

```text
L_market =
    lambda_text       * L_text_generation
  + lambda_structured * L_structured_state
  + lambda_contrast   * L_market_text_contrastive
  + lambda_masked     * L_masked_market_modeling
  + lambda_future     * L_future_market_state
```

其中：

| 损失 | 作用 |
|---|---|
| `L_text_generation` | 生成自然语言市场摘要 |
| `L_structured_state` | 分类或回归结构化市场状态 |
| `L_market_text_contrastive` | 对齐市场 tokens 和对应文本 |
| `L_masked_market_modeling` | 恢复被遮盖的行业、指数或宽度信息 |
| `L_future_market_state` | 保证表征包含与未来相关的信息 |

最终还必须加入主任务的 Pixel Diffusion loss 联合微调。只有文本重建会倾向于保留人类容易描述的信息，而丢弃预测需要的连续细节。

## 10. 股票、行业和概念编码

### 10.1 股票身份

股票条件至少包括：

```text
stable security id
exchange
board
上市时长分桶
市值分桶，可选
```

证券代码不能直接作为永久公司身份，需要通过稳定 security id 处理代码变化、更名、重新上市和退市。

### 10.2 层级行业

行业编码保留层级：

```text
Industry Level 1 Token
Industry Level 2 Token
Industry Level 3 Token
```

行业标签必须包含：

```text
security_id
industry_standard
industry_level
industry_id
valid_from
valid_to
```

### 10.3 概念集合

概念是无序、多标签且会变化的集合，不使用字符串顺序作为位置关系：

```text
Concept Embeddings
  -> Set Transformer / Attention Pooling
  -> 2～4 个 Concept Summary Tokens
```

概念元数据同样必须按历史有效期读取。未知概念、缺失概念和概念数量过多需要稳定的 mask 与截断策略。

## 11. Multimodal Fusion Encoder

### 11.1 Context Token 序列

不同模态先在各自模块内编码，再组成统一 context：

```text
[KLINE_VIS_1 ... KLINE_VIS_N]
[MARKET_SEM_1 ... MARKET_SEM_S]
[MARKET_NUM_1 ... MARKET_NUM_Q]
[ASSET]
[INDUSTRY_L1 INDUSTRY_L2 INDUSTRY_L3]
[CONCEPT_1 ... CONCEPT_K]
[STOCK_SCALE_1 ... STOCK_SCALE_R]
```

每个 token 叠加：

```text
modality embedding
token position embedding
time embedding, if applicable
shared projection to d_model
```

### 11.2 融合机制

第一版采用 Fusion Transformer 建立跨模态关系：

```text
All Condition Tokens
  -> Fusion Transformer
  -> Context Tokens
```

Pixel Diffusion Decoder 通过 cross-attention读取 Context Tokens。不要把所有模态仅平均成一个向量，否则会过早形成信息瓶颈，也无法观察不同条件对生成路径的注意力关系。

### 11.3 条件缺失

系统需要支持部分条件缺失，以便推理、消融和 classifier-free guidance：

```text
<NULL_MARKET>
<NULL_INDUSTRY>
<NULL_CONCEPT>
<NULL_ASSET>
```

训练时按配置随机丢弃部分条件。Diffusion 推理时可使用有条件和无条件预测组合控制条件强度。

## 12. Pixel-space Diffusion Decoder

### 12.1 输出定义

Diffusion 直接生成固定尺寸的未来路径画布：

```text
输入: 带噪路径图 x_tau、diffusion timestep tau、Context Tokens
输出: pixel-space noise / velocity / clean image prediction
目标: 未来 30 日累计收益路径图
```

不经过历史 K 线 image tokenizer 的 decoder，也不把未来路径先编码为 latent。

### 12.2 固定坐标系

输出图必须使用不依赖未来信息的固定坐标：

```text
横轴: 未来第 1～30 个交易日
纵轴: 标准化累计对数收益
零轴: y = 0
```

纵轴可以使用全局固定收益区间，或使用观测时点已知的历史波动率进行标准化：

```text
z_k = y_k / sigma_t
```

其中 `sigma_t` 只能由 `t` 日及以前计算。输出画布需要固定裁剪区间，并额外记录超界 mask 或 tail bucket，不能根据未来路径自动缩放。

### 12.3 路径栅格化

单像素折线对抗锯齿和轻微位移过于敏感。建议将目标路径渲染为窄概率带：

```text
每个未来日对应固定 x 坐标
真实累计收益映射为中心 y 坐标
中心附近像素按高斯核或距离衰减赋值
零轴、边界和路径使用独立通道或稳定编码
```

一个可行的输出张量为：

```text
height x width x channels

channel 1: cumulative return path density
channel 2: valid horizon mask
channel 3: optional uncertainty / scale target
```

第一版是否使用单通道或多通道应通过实验确定，但渲染与反解必须是确定性的。

### 12.4 从像素恢复路径

对第 `k` 个交易日所在列，从像素密度恢复纵坐标：

```text
p_k(y) = normalize(image[y, x_k])
y_hat_k = sum_y p_k(y) * coordinate(y)
```

然后反标准化为累计收益：

```text
y_hat_k = z_hat_k * sigma_t
```

除期望路径外，还可从像素列分布或多次 diffusion 采样得到分位数带。

### 12.5 Diffusion 网络

Pixel-space Decoder 可选 2D U-Net 或小型 pixel DiT：

```text
Noisy Path Image
+ Diffusion Timestep Embedding
+ Cross-attention Context
  -> Pixel Diffusion Network
  -> Pixel Prediction
```

由于画布远小于自然图像，第一版不需要沿用大规模文生图网络。优先选择结构简单、能稳定读取 Context Tokens 的条件 U-Net 或 DiT。

### 12.6 路径约束损失

仅使用像素 diffusion loss 可能生成视觉合理但数值偏移的路径。建议增加从预测图可微提取的辅助损失：

```text
L_path =
    L_diffusion_pixel
  + lambda_coord    * L_coordinate
  + lambda_terminal * L_terminal_return
  + lambda_drawdown * L_max_drawdown
  + lambda_smooth   * L_optional_geometry
```

`L_optional_geometry` 不能强制路径过度平滑，只用于抑制纯像素噪声造成的非金融锯齿，并需通过消融验证。

## 13. 多任务 Decoder

共享 Encoder 后可连接多个 Decoder：

```text
Fusion Context
  +-> Pixel Diffusion: 30 日累计收益路径
  +-> Horizon Return Head: 1 / 5 / 10 / 20 / 30 日收益
  +-> Risk Head: 波动率、最大回撤、最大上涨
  +-> Relative Return Head: 相对行业和相对基准收益
  +-> Regime Head: 当前市场状态
  +-> Reconstruction Head: 遮盖历史信息恢复
```

第一版只保留主任务和三个辅助任务：

1. 5 / 20 / 30 日累计收益。
2. 未来 30 日最大回撤。
3. 未来 30 日相对行业收益。

多任务总损失：

```text
L_total =
    L_path
  + lambda_horizon * L_horizon_return
  + lambda_risk    * L_risk
  + lambda_relative * L_relative_return
  + lambda_market  * L_market_pretraining
```

辅助任务每增加一项都必须进行单独消融，避免任务冲突损害路径生成。后续可使用可学习损失权重或梯度冲突处理，但第一版不提前增加复杂训练实体。

## 14. 训练阶段

### 14.1 Stage A: 数据和渲染器验证

目标：确保相同原始数据始终生成相同输入图和目标路径图。

验收：

```text
复权正确
时间边界正确
行业概念历史有效性正确
路径图可无歧义恢复数值路径
训练集统计量不读取验证和测试数据
```

### 14.2 Stage B: Market Encoder 语义预训练

```text
Market Info
  -> Market Encoder
  -> Structured State / Text / Contrastive Objectives
```

先建立市场状态压缩能力和语言语义，再加入最终个股路径任务。

### 14.3 Stage C: 多模态 Encoder 监督预训练

冻结通用 Image Tokenizer，训练：

```text
Image Projector
Market Encoder
Meta Encoder
Fusion Encoder
Horizon / Risk / Relative-return Heads
```

该阶段用低成本监督任务确认各模态确实能提供样本外增益。

### 14.4 Stage D: Pixel Diffusion 训练

先冻结或低学习率更新 Encoder，训练 Pixel Diffusion。待路径生成稳定后，再对 Fusion Encoder 和 Market Encoder 进行小学习率联合微调。

### 14.5 Stage E: 多任务联合微调

最终联合优化主任务和保留的辅助任务。训练日志必须分别记录每个损失、梯度范数和验证指标，不能只记录总损失。

## 15. 数据集和缓存

### 15.1 样本逻辑结构

```text
sample_id
security_id
observation_date
history_image_ref
image_token_ref, optional cache
market_daily_token_refs
industry_ids_with_validity
concept_ids_with_validity
stock_scale_features
future_cumulative_log_returns[30]
future_path_image_ref
future_mask[30]
```

### 15.2 中间缓存

为控制训练成本，建议缓存：

```text
历史 K 线图像
冻结 tokenizer 输出的视觉 tokens
Daily Market Tokens
结构化市场文本标签
未来路径图
```

缓存键必须包含数据版本、渲染器版本、tokenizer 版本、特征 schema 版本和观测截止日期，避免静默复用过期表示。

### 15.3 数据切分

禁止随机拆分股票日期样本。建议严格按时间切分：

```text
Train:      较早连续年份
Validation: 后续连续时间段
Test:       最后完全隔离时间段
```

最终评估采用滚动或 expanding-window walk-forward。相邻切分边界需要考虑 30 日标签重叠，使用 purge gap 防止未来标签跨越边界。

### 15.4 股票池

每个历史日期使用当时真实可交易股票池，包含后来退市股票，并正确处理：

```text
IPO 冷启动期
停牌
涨跌停
ST / 风险警示
交易板块规则
缺失未来 30 日标签
退市前样本
```

## 16. 评估体系

### 16.1 图像和路径指标

```text
Pixel reconstruction / denoising loss
路径坐标 MAE / RMSE
终点 30 日累计收益误差
1 / 5 / 10 / 20 / 30 日路径误差
最大回撤误差
最大上涨误差
方向准确率
路径相关系数
```

### 16.2 分布质量

Diffusion 的价值在于生成条件分布，必须评估：

```text
真实路径在预测分位数带内的覆盖率
预测区间宽度
终点收益 CRPS 或其他 proper scoring rule
多次采样的路径多样性
不同市场 regime 下的条件校准
尾部亏损覆盖率
```

只比较单条均值路径会忽略 diffusion 的主要价值。

### 16.3 横截面指标

```text
Rank IC
ICIR
相对行业 Rank IC
Top-K / Bottom-K 收益差
不同市值、行业和市场状态下的稳定性
```

### 16.4 交易相关评估

生成模型不能只用回测收益评价，但可以加入隔离的策略评估：

```text
使用固定、简单、预先定义的路径打分规则
包含手续费、滑点、涨跌停和停牌约束
禁止根据测试结果反复调整策略阈值
报告换手率、容量和最大回撤
```

## 17. 消融实验

至少执行以下实验矩阵：

| 实验 | 目的 |
|---|---|
| K 线图像 vs 等价数值 OHLCV Encoder | 验证图像路线本身的增益 |
| 无 Market Encoder | 测量全局市场条件的边际价值 |
| 仅当天市场 vs 市场历史序列 | 测量 Temporal Market Encoder 的价值 |
| 仅 Semantic Tokens | 检查语言瓶颈造成的信息损失 |
| 仅 Quantitative Tokens | 测量 VLM 语义对齐的增益 |
| Semantic + Quantitative Tokens | 验证双通路方案 |
| 无文本对齐 | 判断 LLM/VLM 预训练是否改善主任务 |
| 64 / 48 / 32 / 16 Market Tokens | 确定压缩率 |
| 无行业概念条件 | 测量元数据贡献 |
| Diffusion vs 确定性路径回归 | 判断分布建模是否必要 |
| Pixel Diffusion vs 其他连续生成基线 | 验证 pixel-space 代价与收益 |
| 单任务 vs 多任务 | 检查共享表征是否增强 |

所有消融必须使用完全相同的数据切分和评估脚本。

## 18. 风险与待决策项

### 18.1 通用 Image Tokenizer 的领域差异

通用 tokenizer 可能更关注颜色、边缘和纹理，而不是金融尺度。当前设计接受通用 tokenizer 作为前提，但需要通过冻结与微调消融验证其信息保留程度。

### 18.2 文本语义瓶颈

VLM 训练可能让 Market Encoder 只保留容易描述的强语义，丢弃微弱连续信号。Semantic 和 Quantitative 双 query 是当前缓解方案，是否需要完全独立的分支由实验决定。

### 18.3 极高压缩率

Market Tokens 太少会丢失行业轮动、尾部和相关性结构；太多则增加计算并可能记忆噪声。压缩率必须按主任务、文本任务和校准指标共同选择。

### 18.4 Pixel Diffusion 的稀疏目标

未来路径图比自然图像稀疏，标准像素噪声目标可能效率不高。需要比较单像素线、概率带、多通道图和不同噪声调度，但不改变 pixel-space 生成前提。

### 18.5 生成图与数值路径不一致

必须使用确定性的 rasterizer 和 differentiable extractor，并同时评估像素和路径坐标。视觉质量不能替代数值正确性。

### 18.6 数据偏差

主要风险包括幸存者偏差、复权错误、行业概念回填、停牌处理、IPO 样本和未来标签重叠。数据验证优先级高于模型扩容。

### 18.7 多任务冲突

辅助任务可能强化市场 beta，却损害个股相对收益表征。每个 Decoder 必须有消融证据后才能保留。

### 18.8 仍需实验确定的参数

```text
历史 K 线窗口长度和图像尺寸
通用 image tokenizer 型号和 token 数
Market Encoder 历史窗口长度
Daily / Semantic / Quantitative token 数
Market Encoder 使用 Q-Former 还是 Perceiver Resampler
Fusion Transformer 深度和宽度
路径画布尺寸和纵轴标准化方式
Pixel U-Net 还是 Pixel DiT
diffusion prediction target 和采样器
classifier-free guidance 强度
辅助任务损失权重
```

## 19. 分阶段实施路线

### Phase 0: 数据与基线

1. 固定样本时点和历史股票池。
2. 实现确定性的历史 K 线图和未来路径图数据集。
3. 建立数值 OHLCV Encoder + 确定性路径回归基线。
4. 建立无 Market Encoder 的图像条件基线。

### Phase 1: 最小可行 World Model

```text
历史日 K 图像
  -> 通用 Image Tokenizer
  -> Visual Projector

基础市场指标历史
  -> 小型 Temporal Market Encoder

股票 ID + 三级行业
  -> Meta Tokens

Fusion Encoder
  -> Pixel Diffusion
  -> 未来 30 日累计收益路径图
```

第一阶段不加入全市场股票级 Set Encoder、概念集合和 LLM。

### Phase 2: 分层 Market Encoder

1. 加入行业聚合。
2. 加入市场宽度和风格状态。
3. 缓存 Daily Market Tokens。
4. 增加 Semantic / Quantitative 双 query。
5. 完成不同压缩率消融。

### Phase 3: VLM 语义对齐

1. 自动生成结构化市场状态。
2. 生成受约束自然语言市场摘要。
3. 训练 Market-Text contrastive 和 text generation。
4. 对比有无文本对齐的主任务表现。

### Phase 4: 完整元数据与多任务

1. 加入历史有效概念集合。
2. 增加相对行业收益和风险 Decoder。
3. 加入条件缺失训练和 classifier-free guidance。
4. 进行端到端联合微调。

### Phase 5: 扩展模态

仅在前述模块有明确样本外收益后，考虑加入：

```text
分钟 K 线图像
指数和行业市场图像
公告和新闻文本
财务报表
资金流
期货、利率、汇率和商品
持仓与交易 action
```

## 20. 第一版建议配置

第一版建议控制复杂度如下：

```text
预测频率: 日频
历史窗口: 256 个交易日
预测 horizon: 30 个交易日
历史输入: 单张日 K 图像，包含成交量
Image Tokenizer: 通用 tokenizer，冻结
Market History: 60 个交易日
Market Input: 主要指数 + 市场宽度 + 行业聚合
Market Tokens: 16 semantic + 32 quantitative
Meta Input: security id + exchange + 三级行业
Concept Tokens: 第一版关闭
Fusion: 中小型 Transformer
Decoder: 条件 Pixel-space Diffusion
主输出: 30 日累计对数收益路径图
辅助输出: 5/20/30 日收益、30 日最大回撤、相对行业收益
```

第一版的核心验收问题不是生成图是否美观，而是：

1. 图像条件是否优于等价数值基线。
2. Market Encoder 是否提供稳定的样本外增益。
3. Semantic + Quantitative 双 tokens 是否优于单一语义压缩。
4. Diffusion 样本分布是否校准。
5. Pixel 图能否稳定、无偏地还原为数值路径。

## 21. 设计原则总结

1. 历史 K 线使用通用 image tokenizer，但不假设 tokenizer 已经具备金融预测语义。
2. 未来路径由 Pixel-space Diffusion 直接生成，不经过 latent decoder。
3. Market Encoder 采用横截面、时间和 query 压缩的分层结构。
4. 文本对齐提供语义，不作为唯一信息通道。
5. Semantic Tokens 和 Quantitative Tokens 共同服务最终路径生成。
6. 行业和概念按历史有效期编码，概念作为无序集合处理。
7. 所有模态以 Context Tokens 融合，避免过早压缩为单向量。
8. 多任务 Decoder 只有在消融证明有效后才保留。
9. 评估同时关注数值路径、概率校准、横截面能力和交易约束。
10. 优先完成最小闭环，再逐步增加全市场截面、VLM 和更多模态。
