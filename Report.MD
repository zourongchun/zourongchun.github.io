分析完成。目录已命名 qa_bs512_max256_dense_off_12,yaml 已翻回 ON。9d774e6 官方口径 ON/OFF 终评(on14 vs off12,同代码同配置)：
指标 (r0)	OFF (off_12)	ON (on_14)
equivalent_latency_ms	60.80 (r1 61.77)	63.93 (r1 64.26)
有效 decode TPS	466.7 tok/s	441.5
含尾部空跑 decode TPS	190.4	212.2
全局 TPS (16 rank)	7199 / 3049	6891 / 3398
model_only_step_ms	116.5	99.6
framework_overhead_step_ms	56.1	67.9
avg_delivered_length	2.839	2.620
关键发现——尾部空跑是 ON 全程收益的唯一来源：
- 尾部(155/146 步，占 57-60%):OFF 跑静态 w6 宽图 162.3ms/步，ON 空跑段自动降为 seed-only 窄图 135.9ms/步 → 省 4.35s;active 阶段 ON 反而慢 1.0s
- 所以 ON 的 +11.4% 全程 TPS 是调度器行为红利(空跑到 max_new_tokens),不是规划质量收益；OFF 若同样跳尾部即被中和
- 请求侧结论与 §17.3 推导口径一致且收敛(+5.1% vs +8.0%):静态 L=5 仍接近最优，ON 输在行级置信度裁剪(交付 2.84→2.62)+ 调度税 11.8ms/步


分析完毕，规划等开销(framework_overhead)的构成与优化点：
逐项分解(ms/步，r0):
组件	旧线 off_10	off_12	on_14
model_input	2.57	32.74	36.44
cache_mapping	2.01	32.05	29.62
plan_search	0	0	6.02
cost 同步路径	—	0	4.5+1.3
proposal_sampling_input	0.40	0.40	4.99
合计(wall−model)	19.5	54.9	67.2
两处 f1d8914 硬回退(占 60ms,双侧受害，是 P0):
1. cache_utils.py:548 prepare_slot_mapping——88c758b 已向量化(repeat_interleave 一次建索引、仅 1 次 .item()),f1d8914 整个 revert 回逐请求循环：32 行 × 2 次 .item() 同步 + 逐行 gather → cache_mapping 2→32ms
2. execution_engine.py npugraph_ex 分支——88c758b 只 1 次 D2H(query lengths 直接取 plan 的 CPU 副本)，f1d8914 改成每步 3 次 .cpu().tolist() + 新增 2-3 处 accepted_num.cpu().tolist() → model_input 2.6→32.7ms
3. 这两处与动态规划无关，是“isolate planning path”重构误伤；9d774e6 没修，OFF 同样 +60ms(off_12 framework 54.9 vs 旧线 19.5)
ON 专属税 ~12ms(P1):plan_search 6ms(决策可跨步缓存，verification_plan.py:586)+ cost 路径 5ms(校准修复后病根已除，可安全恢复 88c758b 的异步 overlap)。P2:proposal_sampling_input 4.6ms、verify 5-6ms。
修复后投影:P0 revert → OFF 墙钟 171→111ms(equivalent_latency 60.8→*39.6**),ON 166→106(63.9→*40.5*)——*ON/OFF 差距从 +5.1% 缩到 ~+2%,当前对比结论被公共回退严重失真**。


framework 细粒度分解(on16 vs off13,每活跃步；嵌套已按 execution_engine.py:974-1235 核对，账面闭合)*：
层级	on_16	off_13	Δ
input_preparation	40.47	32.93	+7.54
├ plan_and_pack(plan_search)	7.21(6.89)	0.26(0)	+6.95
├ cache_mapping	31.87	31.64	+0.23(双侧公共，§17.5)
└ batch_tensor+misc	1.39	1.03	+0.36
proposal 非模型开销	10.30	5.20	+5.10
├ proposal_sampling_input	4.46	0.18	+4.28
├ graph_copy/input/state 等	2.05	1.51	+0.54+…
gap(未记账)	12.56	10.56	+2.00
verify	5.64	5.96	−0.32
合计	68.97	54.65	+14.32
劣化排序：① plan_search +6.89(48%)=cpu 3.30+collective 1.42+d2h 1.42+to_host 1.02+prepare 0.52;② *proposal_sampling_input +4.28(30%)*——常量标志的每步 all_reduce 协调，30ca40e 只修掉一小半(OFF 侧仅 0.18);③ gap +2.00(未记账，需 profiler);④ 杂项 ~+1.4。
两个新发现：搜索缓存完全无效(hits=0 / misses=52000≈512行×101步——键粒度逐行，每步全变，需行组级键)；异步重叠闲置(cost_async_total=0——6.9ms 全串行在关键路径，异步化空间完整)。
