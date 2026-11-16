11月16日：
off29(动态关)vs onzos2(动态开+ZOS)逐模块对比(ms/step)
模块(嵌套关系)	OFF	ON+ZOS
step 合计	169.15	167.08
输入准备	30.00	31.20
├ model_input(含采样协调)	29.44	30.38
│  └ plan_and_pack(规划)	0.24	0.28
├ metadata	0.33	0.42
└ cache_mapping	28.82	29.62
Target 调用	111.02	96.57
├ target_model	102.36	85.63
└ verify	5.63	5.18
Proposal 窗口(提交→发布)	19.30	33.04
├ 设备 draft(model_wall)	16.25	23.96
│  └ 窗口内： 预取 next_plan_prepare	0	18.05
│  └ 窗口内： host 簿记	0	2.88
├ state_update	0.39	0.66
框架 gap(未细分部分)	~8.8	~6.3
accept(每轮交付 token)	2.817	2.625
eq = step / accept	60.05	63.65
因果链
1. step 时间 ON 反而快 2.1ms:Target 收益 −14.5ms(L3 验证宽度 4 vs OFF 固定 L5 宽度 6,验证图更小)几乎被 Proposal 窗口 +13.7ms 抵消——窗口内塞入 18.05ms 预取，超过其想隐藏的 ~16ms 设备 draft 时间，设备尾部空转 ~8ms
2. eq 反而差 6% 的决定因素是 accept 掉 6.8%(2.617→2.625?不对：2.817→2.625):动态规划选 L3 每轮交付 2.625,OFF 固定 L5 交付 2.817——每 token 成本被摊薄得更少
3. 也就是说：动态 verify 开+ZOS 的差距不在某个 host 耗时块，而在“L3 vs L5 的接受长度损失”；时间块层面，预取撑大的 Proposal 窗口(+13.7)吃掉了小验证宽度收益(−14.5),使 step 的净优势只剩 2.1ms,不足以覆盖 accept 损失
若要追平 OFF:要么把预取压进 16ms 设备窗口内(step 再省 ~5ms,仍不够单独翻盘)，要么提高 L3 的 accept(决策预算向接受长度倾斜)——后者杠杆更大(accept 每提高 0.1,eq 降约 2.4ms)。


1116_1546:
GAP 1:每请求 .item() 同步循环(defer 路径下击穿重叠)
- 位置:executor/core/forward_data_info/forward_data_info.py:445-484(update_requests_from_batch 循环)，:451 state.accepted_num.item();同类还有 ：488-493 update_speculative_state_from_batch→:151
- 证据:profiler 实测 529 次/步 aten::item;defer 分支(execution_engine.py:1290-1300)把循环放进 proposal 执行期间，每次 .item() 等流 drain → 设备 burst 86-90% 饥饿(忙 6.5-8.7ms / wall 55-63ms);窗口 host 3.22→11.0ms(+7.76)
- 改法：循环外一次批量读回:accepted_num.tolist()(computed_lens 同理)，或把簿记挪回流空闲时段
- 预期:proposal_total 24.75 → 18ms,step −67ms
GAP 2:plan 应用小算子风暴(动态独有，静态 L 无此开销)
- 位置:proposal 提交路径的 plan take/remap/budget clamp/confidence→seqlens/cache mapping
- 证据：每步 ~1137 个微小设备算子(Gather×96、FloorDiv×106、Add×103、Remainder/FloorMod×100,零模型算子)从 Python 逐个发射
- 改法：整批 device 化——融合成 1-2 个算子，或并入 graph envelope
- 预期：与 GAP 1 合计，proposal 窗口回到甚至低于 OFF 的 19.18(设备 draft 本身 13.77 已比 OFF 快 2.2)
GAP 3:签名协调纯开销(消融已验证)
- 位置:dspark_worker.py:1103(调用点)/:1318(_coordinate_proposal_signature):跨 DP all_reduce + flags.cpu() 两次设备同步
- 证据：消融注释后 eq 63.945→61.656(−2.29ms),accept/delivered 无损；本部署各 DP 签名天然一致，4.96ms/步是纯税
- 改法：签名决策缓存(请求集/采样配置不变时跳过)，或挪进 Target 窗口异步
- 预期：已实测 −2.29ms
汇总
 	eq TPOT
OFF 基线	60.94
ON+ZOS 现状	61.66(+1.2%)
GAP 1+2+3 全修后预估	~58.5(−4%,反超 OFF)
前提：accept(2.64)不变；若 GAP 1/2/3 修完仍无收益，剩余就是 accept 结构性差距(决策预算倾斜问题，非代码税)。
