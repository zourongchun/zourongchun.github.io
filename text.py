# coding=utf-8
# Adapted from benchmark/evalscope_scripts/eval_longbench.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EvalScope 自定义模型评测脚本，支持投机解码接受率统计。

在 eval_longbench.py 基础上增加：
  1. 透传 infer 仓 server /v1/chat/completions 响应 usage 中的 spec_* 字段
     到 ModelOutput.metadata，使 EvalScope prediction 产物保留 per-request
     接受率/接受长度。
  2. 模块级累加 spec 统计，评测结束后打印汇总报告（接受率/平均接受长度）。
  3. --datasets 参数化，支持 math_500 / humaneval / mmlu / longbench_v2 等。

spec 字段来源：executor/online/server.py:_usage (line 438-446)
  spec_acceptance_rate      = accepted / draft
  spec_num_accepted_tokens  累计接受 draft token 数
  spec_num_draft_tokens     累计提议 draft token 数
  spec_num_forward_ct       累计验证步数
  spec_avg_accept_length    = accepted / forward_ct + 1

聚合公式（与 server.py:360,361 一致）:
  acceptance_rate    = Σaccepted / Σdraft
  mean_accept_length = 1 + Σaccepted / Σforward_ct

用法:
  python eval_with_spec_stats.py \
      --model-name deepseek_v4 \
      --base-url http://localhost:8000/v1/chat/completions \
      --model-path /data/models/deepseek_v4 \
      --datasets math_500 \
      --max-tokens 8192 \
      --batch-size 8

  # 长上下文（自动中间截断）
  python eval_with_spec_stats.py --datasets longbench_v2 --max-prefill-tokens 102400
"""

import logging
import json
import argparse
import threading
from typing import List, Optional, Dict, Any, Tuple

import requests
from evalscope import run_task
from evalscope.config import TaskConfig

from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAFETY_MARGIN = 4 * 1024

SPEC_FIELDS = [
    "spec_acceptance_rate",
    "spec_num_accepted_tokens",
    "spec_num_draft_tokens",
    "spec_num_forward_ct",
    "spec_avg_accept_length",
]


class SpecStatsAccumulator:
    """线程安全的投机解码统计累加器。EvalScope 内部可能并发调用 generate。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.accepted = 0
        self.draft = 0
        self.forward_ct = 0
        self.num_requests_with_spec = 0
        self.num_requests_total = 0

    def add(self, usage: Dict[str, Any]) -> None:
        self._lock.acquire()
        try:
            self.num_requests_total += 1
            if "spec_num_draft_tokens" not in usage:
                return
            self.num_requests_with_spec += 1
            self.accepted += int(usage.get("spec_num_accepted_tokens", 0))
            self.draft += int(usage.get("spec_num_draft_tokens", 0))
            self.forward_ct += int(usage.get("spec_num_forward_ct", 0))
        finally:
            self._lock.release()

    def report(self) -> Dict[str, Any]:
        with self._lock:
            acceptance_rate = self.accepted / self.draft if self.draft > 0 else 0.0
            mean_accept_length = 1 + self.accepted / self.forward_ct if self.forward_ct > 0 else 1.0
            return {
                "num_requests_total": self.num_requests_total,
                "num_requests_with_spec": self.num_requests_with_spec,
                "spec_num_accepted_tokens": self.accepted,
                "spec_num_draft_tokens": self.draft,
                "spec_num_forward_ct": self.forward_ct,
                "acceptance_rate": acceptance_rate,
                "mean_accept_length": mean_accept_length,
            }


_spec_accum = SpecStatsAccumulator()


class SpecAwareModel(ModelAPI):
    """自定义模型实现，透传投机解码统计到 ModelOutput.metadata。"""

    def __init__(
            self,
            model_name: str,
            max_prefill_tokens: int,
            max_tokens: int,
            timeout: int,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            pretrained_model_name_or_path: Optional[str] = None,
            config: Optional[GenerateConfig] = None,
            **model_args: Dict[str, Any],
    ) -> None:
        if config is None:
            config = GenerateConfig()
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        self.max_prefill_tokens = max_prefill_tokens
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model_name = model_name

        if pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True
            )
        else:
            self.tokenizer = None

    def generate(
            self,
            input: List[ChatMessage],
            tools: List[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
    ) -> ModelOutput:
        input_text = self._process_messages(input)
        content, usage = self._call_model(input_text, config)

        _spec_accum.add(usage)

        spec_meta = {k: usage.get(k) for k in SPEC_FIELDS if k in usage}
        metadata = {"spec_decoding": spec_meta} if spec_meta else None

        output = ModelOutput.from_content(
            model=self.model_name,
            content=content,
        )
        output.metadata = metadata
        return output

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        text_parts = []
        for message in messages:
            role = getattr(message, 'role', 'user')
            content = getattr(message, 'content', str(message))
            content = self.middle_truncation(content)
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)

    def middle_truncation(self, prompt: str) -> str:
        if self.tokenizer is None:
            return prompt
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) <= self.max_prefill_tokens:
            return prompt

        global_budget = self.max_prefill_tokens - SAFETY_MARGIN
        input_ids = input_ids[:global_budget // 2] + input_ids[-global_budget // 2:]
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return prompt

    def _call_model(self, input_text: str, config: GenerateConfig) -> Tuple[str, Dict[str, Any]]:
        try:
            content, usage = self.request_ai(input_text)
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败 (URL: {self.base_url}): {e}")
            raise RuntimeError(f"Model API request failed due to network issue: {e}") from e
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"接口返回格式异常，解析失败: {e}")
            raise RuntimeError(f"Model API response parsing failed: {e}") from e

        return content, usage

    def request_ai(self, content: str) -> Tuple[str, Dict[str, Any]]:
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": self.max_tokens
        })

        headers = {
            "Content-Type": "application/json"
        }
        response = requests.request("POST", self.base_url, headers=headers, data=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {}) or {}
        return text, usage


def print_spec_report(stats: Dict[str, Any]) -> None:
    has_spec = stats["num_requests_with_spec"] > 0
    print("=" * 60)
    print("Speculative Decoding 接受率汇总（本次 benchmark）")
    print("=" * 60)
    print(f"请求总数           num_requests_total     : {stats['num_requests_total']}")
    print(f"含 spec 统计请求数 num_requests_with_spec : {stats['num_requests_with_spec']}")
    if not has_spec:
        print("[warn] 无 spec 统计，请确认 server 开启了投机解码 "
              "(speculative_config.num_speculative_tokens > 0)")
        print("=" * 60)
        return
    print("-" * 60)
    print(f"累计接受 draft  spec_num_accepted_tokens : {stats['spec_num_accepted_tokens']}")
    print(f"累计提议 draft  spec_num_draft_tokens    : {stats['spec_num_draft_tokens']}")
    print(f"累计验证步数    spec_num_forward_ct      : {stats['spec_num_forward_ct']}")
    print("-" * 60)
    ar = stats["acceptance_rate"]
    print(f"接受率      acceptance_rate = Σacc/Σdraft        : {ar:.4f}  ({ar*100:.2f}%)")
    print(f"平均接受长度 mean_accept_length = 1+Σacc/Σfwd_ct : {stats['mean_accept_length']:.4f}")
    print("=" * 60)


def main(args):
    custom_model = SpecAwareModel(
        model_name=args.model_name,
        base_url=args.base_url,
        pretrained_model_name_or_path=args.model_path,
        max_prefill_tokens=args.max_prefill_tokens,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    task_config = TaskConfig(
        model=custom_model,
        datasets=[args.datasets],
        generation_config={
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "timeout": args.timeout
        },
        eval_batch_size=args.batch_size,
        limit=args.limit,
    )
    if args.dataset_local_path:
        task_config.dataset_args = {args.datasets: {"local_path": args.dataset_local_path}}

    logger.info("开始评测，datasets=%s", args.datasets)
    run_task(task_cfg=task_config)

    stats = _spec_accum.report()
    print_spec_report(stats)

    if args.spec_report:
        with open(args.spec_report, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info("spec 汇总报告已保存 -> %s", args.spec_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行 EvalScope 评测并统计投机解码接受率")

    parser.add_argument("--model-name", type=str, default='deepseek_v4', help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="模型 API 地址（含 /v1/chat/completions 路径）")
    parser.add_argument("--model-path", type=str, default="",
                        help="本地模型权重路径（用于 tokenizer 做中间截断；留空则不截断）")

    parser.add_argument("--datasets", type=str, default="math_500",
                        help="评测数据集，如 math_500 / humaneval / mmlu / longbench_v2")
    parser.add_argument("--dataset-local-path", type=str, default="",
                        help="数据集本地目录（含 test.jsonl / eval.yaml），注入 dataset_args[ds].local_path；留空则从 modelscope 下载")
    parser.add_argument("--max-prefill-tokens", type=int, default=102400, help="最大 Prefill Token 数（中间截断阈值）")
    parser.add_argument("--max-tokens", type=int, default=8192, help="生成最大 Token 数")

    parser.add_argument("--batch-size", type=int, default=8, help="生成与评测的 Batch Size")
    parser.add_argument("--timeout", type=int, default=1200, help="请求超时时间(秒)")
    parser.add_argument("--limit", type=int, default=None, help="限制评测的数据条数（测试用，默认不限制）")
    parser.add_argument("--spec-report", type=str, default="",
                        help="spec 汇总报告 json 输出路径（可选）")

    main(parser.parse_args())
