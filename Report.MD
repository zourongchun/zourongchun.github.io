11月27日：
899830a（P0 kv_len 提前 staging）验证结论
先说阻塞：远端漏加 ZosStepState.wait_host_kv（只加在 ZosCommitResult 上），ON 首跑即 AttributeError 崩溃——本地热修 3 行后完成验证，需远端补齐。
功能面全部通过，但性能面无收益：
每 step (ms)	OFF
wall	65.415
逐项：step1 回退 ✓ / 稳态 staged=True 22/22 ✓ / metadata_ms 0.245→0.109（.cpu() 成功跳过）✓ / 输出 16/16 rank bit-identical ✓ / wall 未达 −4~−5ms 预期 ✗
根因（新瓶颈 P2，代码实锤）：scheduler.py:212 每个 run_step 开头 _drain_zos_deferred_releases()（:608-619 逐个 event.synchronize()），:702 每个请求结束都排队 release（release_event=下一轮 completion event）。只要有请求结束，host 在 prep 之前就被同步到上一轮 device 完成点——P0 删掉的 .cpu() 阻塞被这个 sync 原位顶替，commit+batch+prep+submit ~4-6ms 仍串在轮边界。佐证：ZOS 内部等待全测 ≈0（drain 已先同步）、framework 6.96 与 fd8ea3a 的 6.64 同构、gap 位置时长与 drain+commit+prep 吻合。
另发现：ON 计数 32106 vs OFF 32550（−444）——生成结束 flush 的末轮 token 在计数窗口后落账，输出不受影响，纯口径问题。
P2 建议（反馈远端）：块释放改惰性——请求结束仅 host 退役标记，synchronize()+free 推迟到 KV 池分配需要复用时。纯 decode 无新分配 → 同步完全退出关键路径，预期 wall 65.6 → ~60。
TPS/TPOT 相对 OFF：+0.32% 带内持平，无实质劣化亦无收益；ON 环比（vs 7a791a6 ibench ON 65.979）−0.54%。
