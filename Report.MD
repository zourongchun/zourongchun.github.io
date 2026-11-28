11月28日：
优化方向：把 ZOS 路径上"每步 pageable H2D"全部消除，让 ③ 真正立即返回
根因报告 §四 已列了两条（缓存 / pinned），但按 trace 证据应这样排序：
方向 A（首选）：采样参数跨步缓存 + greedy 短路
- 缓存：decode 期每请求 temperature/top_p/top_k/top_logprobs 恒定 → 首次构建 device 张量，仅请求集变化时更新对应行（sampler.py:482-490，每步 5 个 H2D → 降至 0/签名变化时 1 次）
- greedy 短路：_sample_tokens :345-355 在 all_greedy 时直接 argmax 返回，根本不用这些张量——build_sampling_metadata_async 现在却无条件构建（:482 无 all_greedy 判断）。全 greedy 批可完全不建 tensor，只留 Python 标量 flag
- 代价：缓存失效逻辑（请求完成退出/新请求加入时重建行）
方向 B（结构性，必须做）：统一 pinned + non_blocking 通道
仓库已有 ZosHostTransfer 基建（_stage_zos_host_tensors :1484-1486，现只用于 D2H）——把 ZOS 路径所有每步 H2D 改走 pinned buffer + copy_(non_blocking=True)，源 buffer 由 completion event 回收。清单：
- sampler.py:482-490（temperature/top_p/top_k）
- execution_engine.py:1458-1464（input_ids.to+clone、rows、previous_rows）
- :1472-1480（kv_len.to×2、kv_rows_device、previous_kv_rows）
- :1759（current_kv_len.to）
为什么"只修 sampler 那个"无效（结构性约束）
现在 input prep 的 pageable H2D（:1458-1481）不阻塞，是因为轮首流是空的（上一轮已排空）。P3 一旦让 ③ 立即返回 → host 领先 → 下一轮 _apply_zos_device_state 在上轮 53ms 计算仍在流里时执行 → 它的 pageable H2D 成为新的全轮阻塞。所以正确顺序是：先把清单内 H2D 全部转 pinned/缓存，再放开 ③（这也是报告 §三-2 的结论）。
预期收益与验证判据
- span 65.18 → *55-57ms（host 段 10.9ms 藏进上一轮 53ms 计算，只剩 max(device 53.5, host~12) + 衔接）；TPS +1518%*；span 内 12.2ms device idle 与跨轮 1.55ms idle 消失
- 收益上限 = host 段可隐藏量；若要更大，需继续压 host 段本身（10.9ms）
- 复测判据（重跑 zosib2rs 对）：① span 内 aten::to/aclrtSynchronizeStream 消失；② span 中位 → ~55-57ms；③ 跨轮 device idle → ~0；④ 输出 32/32 一致
不建议的偷懒法：把 metadata 构建前移到 replay 之前——流空闲时确实便宜，但 host 一旦领先，轮首窗口也不再空闲，等于没修。缓存（A）+ pinned（B）才是对"host 领先"regime 免疫的解。
