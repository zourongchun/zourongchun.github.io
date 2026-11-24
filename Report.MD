11月24日：
分析结论：这次是新的第二层 bug，上游修复（82ca6ec）其实已生效
证据链：
项	上一轮崩溃（54c3a81）	本轮崩溃（82ca6ec）
崩溃步数	~129 步（首个预留块边界）	第 257 次 decode forward（日志 256 条计时后）
越界请求	prompt 8320（65 块+1 预留）	prompt 8192（截断上限）
表宽/越界	66 / index 66	66 / index 66（同一个数，来源不同）
本轮机制（三环闭合）：
1. max_total_len = input_truncated_len + max_new_tokens×(next_n+1) = 8192+256 = 8448 = 66×128（execution_engine.py:81）——KV 分配上限封顶在 66 块（allocate_slots 里 min(..., max_model_len)，kv_cache_manager.py:53-56），第 67 块永远不会分配
2. prompt=8192 的请求：64 块 prompt + 2 块 decode（位置 8320-8447，恰好 256 个 token）= 表宽 66 ✓
3. max_new=256，本不该存在第 257 轮——但 ZOS 延迟提交使结束判定读到的是“已提交”状态：第 257 轮调度时只看到 255 个已提交 token → 判定未结束 → 多跑一轮 → 写位置 8448 → 需要第 67 块 → 越界崩溃
更严重的伴生问题（不只是 +1 轮的边界问题）：
- ON 臂 batch 从不收缩：trace 显示 rank0 的 32 个请求（0-31 号）从 step 1 到 step 250 恒满（n_req=32 不变）
- OFF 同批请求：平均输出 566 字符、14/32 短于 200 字符——大量 EOS 提前结束
- 即 ZOS 路径下 EOS/结束判定整体失效，512 请求全部被拖到 256 步上限（若不崩将产生 ~13 万 token vs OFF 的 8013）——ON/OFF 完全不可比
两个候选根因（需上游确认，崩溃前未落盘无法直接比对文本）：
1. finish 判定未计入 in-flight token：scheduler.py:529 已算出 inflight_tokens 但只用于 KV 分配，max_new/EOS 终结仍读滞后一轮的 output_id_list；修复方向：len(output) + inflight >= max_new 提前终结
2. 若 commit 回来的 token 流本身有错（_apply_zos_device_state 行重映射错误导致输出偏离 OFF），则永远等不到 EOS——需要远端在 flush 落盘后比对 ON/OFF 文本确认
建议：把上述证据反馈给远端（重点：82ca6ec 已通过 129 步边界 ✓，新问题在请求生命周期终结链路）；OFF 基线数据有效保留（224.35 ms/step，model_only 183.2 / framework 41.1，8013 tokens，255 步 57.2s），等修复后重跑 ON 即可对比。
