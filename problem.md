# DSPark 动态 Verify 问题清单与优化指引

日期：2026-10-16 · 分支：`feat/dspark-dynamic-verify`（`f643f3a`，基线上游 `66236fb`）
硬件：16 × Ascend910（A3，64GB）· 容器 `zrc_infer_dspark`（CANN 9.2.0 / PyTorch 2.9.0 / torch_npu 2.9.0.post7）
模型：`/data/models/deepseek_v4_int8_w8a8_dspark`

> 标注约定：【实测】= 有运行日志/复现证据；【推断】= 有代码证据、待运行验证。

---

## 0. TL;DR

| # | 问题 | 优先级 | 状态 |
|---|---|---|---|
| P0-1 | planner 与 draft 共用同一 `confidence_threshold`，`threshold_prefix` 恒空转 | P0 | 【实测】零裁剪 |
| P0-2 | 上游 gather_v3 索引越界，B>1 必崩（动态 verify 的收益前提被堵死） | P0 | 【实测】上游复现 |
| P1-1 | `warmup_plans()` tnd 布局图数 = (K+1)^B 指数爆炸，B≥3 直接报错 | P1 | 【实测+代码】 |
| P1-2 | 基准方法缺陷：有效 token 仅 29、248/256 步是空转 | P1 | 【实测】 |
| P2 | 编译缓存关闭 / input_prep 开销 / smooth_scales 未加载等 | P2 | 见各条 |

**当前收益结论（B=1）：动态 verify 无净收益。** threshold=0 时 9.69 vs 9.73 ms（持平）；threshold=0.7 时 12.81 vs 13.16 ms（噪声级），且 0.7 本身使两边都变慢 ~32-35%。唯一正面结论：**贪心解码下输出与上游逐字节一致，正确性验证通过**。

---

## 1. 实测数据（temp=0，B=1，npugraph_ex，今日四次 canonical 运行）

| 运行 | 配置 | 等效延迟(ms/token) | accept length | accept rate | 日志 |
|---|---|---|---|---|---|
| 基线 15:19 | 上游代码，threshold 0.0 | 9.73 | 4.833 | 0.885 | worktree `res/20261016/deepseek_v4_deepseek_v4_flash_rank_4_4ep_dspark/log_0.log` |
| 动态 15:37 | 迁移代码，threshold 0.0 | 9.69 | 4.833 | 0.885 | 主仓 `res/20261016/deepseek_v4_deepseek_v4_flash_rank_16_16ep_w8a8_platform/log_0.log` |
| 基线 15:44 | 上游代码，threshold 0.7 | 13.16 | 3.625 | 0.913 | 同基线 |
| 动态 16:01 | 迁移代码，threshold 0.7 | 12.81 | 3.625 | 0.913 | 同动态 |

四条运行 valid/draft/accepted 计数完全对应相等（29/23~26/21~23）——贪心下投机解码输出等价性成立。

---

## 2. 问题详情

### P0-1 planner 与 draft 早停共用阈值 → 阈值裁剪恒为空转【实测】

**证据**：
- `executor/core/model_worker/dspark_worker.py:461`：draft 早停 `confidence.sigmoid() < self.confidence_threshold`
- `executor/core/model_worker/dspark_worker.py:848`：planner 门控 `confident_prefix(planning, self.confidence_threshold)` —— **同一个值**
- threshold=0.7 运行的 256 步逐步指标：凡有 draft 的步 `query_len == draft_len + 1` 恒成立，**裁剪次数 = 0**；与上游基线接受统计逐字节一致

**机理**：draft 早停后幸存 token 置信度必然 ≥ 阈值，再过同阈值的 verify 门恒通过。单阈值双重把关 = 结构性 no-op。

**修复建议**：
- `DynamicVerifyConfig`（`executor/core/config/inference_config.py`）目前**没有** threshold 字段，需新增 `dynamic_verify.verify_threshold`（或 `threshold_margin`）
- `dspark_worker.py:848` 改用独立阈值；draft 侧保持 `confidence_threshold` 不动
- 验证矩阵：draft 0.0 + verify ∈ {0.5, 0.7, 0.9}，确认逐步指标出现 `query_len < draft_len + 1`
- 预期管理：B=1 显存瓶颈下（verify 1 vs 6 token 步耗时仅差 11%：73.3 vs 81.3 ms），裁剪本身收益有限；本步主要目的是**让机制真正运转并验证正确性**

### P0-2 上游 gather_v3 索引越界 → B>1 必崩【实测，与迁移代码无关】

**证据**：
- B=16（batch 256/world 16）时：`GatherV3Base::CheckIdxValue: (0 <= val && val < gxSize_) failed. Index 18 out of range[0 16)`，越界索引 18,24,...,90 = i×6（i=3..15），AIV Block 3-15/16 → 507035 → EE9999
- **上游 `66236fb` worktree + 同配置复现完全相同错误**（`/home/z00976068/infer_dspark_upstream_66236fb`）→ 与迁移代码无关；触发条件是未提交的 YAML 修改（batch 4→256 等，B/rank 1→16）
- 上游原配置 B=1 时 [1,6] 展平偏移仅 {0}，合法 → bug 隐形

**索引签名**：越界值是 [B, K+1]（K+1=6）布局的**展平行偏移 i×(K+1)**，被当作一维索引进了第一维大小为 B 的 gather。B=2 时可预测偏移 {0,6}，6 ≥ 2 仍崩——**B=1 是唯一安全值**。

**待定位**（错误是 barrier 处异步浮出的，具体算子未钉死）。候选（均在 `models/deepseek_v4/models/modeling_dspark.py`，上游文件、迁移未触碰）：
- `_slot_mapping_from_positions`:1086（高级索引 `block_table[rows, block_indices]`，dim-0 恰为 [0,B)）
- `_prepare_dspark_decode_shared_inputs`:1037-1085、`_gather_dspark_kv`:307、`_select_prefill_context_tail`:942、`propose`:1346

**定位方法**：临时把 yaml `exe_mode` 改 `eager`，容器内加 `ASCEND_LAUNCH_BLOCKING=1` 重跑，取精确 Python 行号；修完后记得恢复 npugraph_ex。属上游 bug，修好后建议同步上游。

### P1-1 `warmup_plans()` 图数 (K+1)^B 指数爆炸【实测+代码】

**证据**：`executor/utils/verification_plan.py:276`
```python
count = (self.capacity + 1) ** self.batch_size if self.layout == "tnd" else self.capacity + 1
```
- tnd：B=1 → 6 图（实测 6 × ~54s ≈ 324s 编译；基线同接口单图 62s，每图耗时吻合）；**B=2 → 36 图（~32 分钟）；B=3 → 216 > `max_graphs=64` 直接 ValueError**
- tnd 的 `graph_key`（:113）= `(layout, query_lengths)` 逐请求精确形状；注释明言是刻意规避"backend 只 guard 总 token 数"的假设

**矛盾**：动态 verify 的收益场景是 B>1（compute-bound），而 tnd 精确图撑不到 B>1。

**修复建议**（设计文档本意，`dspark_confidence_dynamic_verify_design.md:231`）：
> 假设只预热了宽度 1、3、6 的图，本轮需要宽度 2 时可能使用宽度 3 的图，补一个 PAD。这叫分桶：用有限规格覆盖动态输入。频繁创建新图可能比节省的 verify 时间更贵。

- B>1 走 **dense 分桶**：`graph_key` 已支持 `(layout, (B, width))`，`build_inputs`（:155）已有 PAD 填充；桶集 {1,3,6} 或由 `cost_profile`（yaml 字段已存在、当前 null）驱动
- 图数从 (K+1)^B 降为常数（K+1 或更少）；tnd 精确图仅保留 B=1
- 顺带解决 warmup 时长：当前 `enable_cache_compile: False`（两个 yaml 都是）导致每次运行全量重编（动态 ~6 分钟 vs 基线 ~1 分钟）

### P1-2 基准方法缺陷【实测】

- 单次运行仅 **29 个有效 token（8 个真实 decode 步）**，248/256 步是请求完成后的空转——1-2ms 级差异无统计意义
- 空转步污染逐步均值（input_prep 12.6ms、step 68.5ms 均含空转）
- 早期对比还出现过 yaml 不一致（eager vs npugraph_ex、mem_fraction 0.85 vs 0.5、multi_streams True/False）

**建议**：
- 换长输出数据集或增大 `max_new_tokens`，保证 ≥500 真实 decode 步
- 统一 yaml 模板：除 `dynamic_verify.enabled` 外全部参数固定（两份 yaml 分别只翻开关）
- 逐步指标分析时按"累计交付 token < valid"截断空转段（本会话已有解析脚本可复用）

### P2 清单

| 项 | 证据/位置 | 说明 |
|---|---|---|
| input_preparation 固定开销 12.6 ms/step（~17% 步耗时） | 逐步指标 | planner 是 CPU 侧逐请求循环，可向量化/缓存；零裁剪时是纯负收益 |
| `enable_cache_compile: True` 未验证对 verification graph 生效 | yaml model_config | 若生效，warmup 324s 可摊薄到首跑 |
| DSpark 权重 19 个 `mtp.*.smooth_scales` 未从 checkpoint 加载 | 动态运行日志 line 137 警告 | draft 量化平滑系数缺失是否影响起草质量，需核对（当前 accept rate 0.88-0.91 看似健康） |
| barrier UserWarning 被编译事件反复重置去重而刷屏（12 vs 4 次） | torch `distributed_c10d.py:4876` | 无害；`init_process_group(device_id=...)` 可消音 |
| rank0 请求结束后空转 248 步陪跑其他 rank | 逐步指标 | 调度层可做 idle 早退/同步收尾优化 |
| 环境版本：CANN 9.2.0 / PT 2.9.0 vs 仓推荐 CANN 9.0.0 / PT 2.8.0 | AGENTS.md | 长期变量，结论异常时回归检查 |

---

## 3. 已验证无需重查的结论

1. **迁移代码门控正确**：`dynamic_verify` 关闭时执行路径与上游等价（静态审计 + 上游 worktree 行为复现双重确认）
2. **输出正确性**：贪心 temp=0、同阈值下与上游逐字节一致（含接受计数）——动态路径未破坏投机解码输出等价性
3. **端到端跑通**：enabled=true / B=1 / npugraph_ex / tnd 下，6 规格 precompile + 图回放全程无错（迁移首次 NPU 验收通过）
4. **gather_v3 与迁移无关**：上游代码同配置必现
5. **编译卡顿定量模型**：324s ÷ 6 图 = 54s/图 ≈ 基线单图 62s；卡顿=编译，非 barrier（基线同 barrier 结构仅 62s；稳态 500+ 次 barrier 无分钟级停顿）
6. **GLIBCXX 修复有效**：容器 `/etc/ld.so.conf.d/00-usr-local-lib64.conf`（指向 /usr/local/lib64）+ ldconfig，**勿删**

---

## 4. 建议实施顺序（依赖链）

```
P0-1 解耦阈值（小改动，机制先转起来）
 └─> P0-2 定位并修复 gather_v3（eager + ASCEND_LAUNCH_BLOCKING=1 定位 → 修 → B=2 回归验证）
      └─> P1-1 dense 分桶 warmup（B>1 的前置）
           └─> 放大 B 验证收益（B∈{2,4}，长输出数据集，P1-2 方法论）
                └─> P2 调优（缓存 / input_prep / cost_profile 桶集选择）
```

B 的配置方式：`batch_size_per_dp_rank = scheduler_config.batch_size // attn_dp_size`（`executor/core/config/inference_config.py:617`）；B=2 即 batch_size 32 / world 16。

---
