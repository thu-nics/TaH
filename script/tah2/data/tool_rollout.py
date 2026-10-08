"""Qwen3 tool rollout: episode-local reference replay, otherwise shape-guided simulation."""

import copy
import json
import re
import threading
import time

import requests

MAX_SEQ_LEN = 16384
THINK_RE = re.compile("^\\s*<think>(.*?)</think>", re.DOTALL)
CALL_RE = re.compile("<tool_call>\\s*(.*?)\\s*</tool_call>", re.DOTALL)
FENCE_RE = re.compile("^\\s*```(?:json)?\\s*(.*?)\\s*```\\s*$", re.DOTALL)
SIM_SYSTEM = """You simulate the backend of a JSON API. You are given one function's schema and one call to it. Return ONLY the JSON body the real API would return.

Rules:
- Output a single JSON object or array. No prose, no markdown fences, no explanation.
- Invent concrete, plausible values that are consistent with the arguments (real-looking names, numbers, dates, ids). Never emit placeholders like "string" or "...".
- If the call cannot be served (bad arguments, missing required field), return an error object such as {"error": "..."}.
- Every identifier, name, place and entity you return must follow from THIS call's arguments. Never carry one over from an example you were shown.
- Keep the body compact: under about 120 words."""


def shape_of(obj, depth: int = 0):
    if depth > 4:
        return "..."
    if isinstance(obj, dict):
        return {k: shape_of(v, depth + 1) for k, v in list(obj.items())[:24]}
    if isinstance(obj, list):
        return [shape_of(obj[0], depth + 1)] if obj else []
    return type(obj).__name__


class Servers:
    def __init__(self, urls: list[str], model: str, timeout: int = 1800):
        self.urls = urls
        self.model = model
        self.timeout = timeout
        self._i = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
            self._local.s.trust_env = False
        return self._local.s

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 0.95,
        top_k: int = 20,
        retries: int = 3,
    ) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        last = None
        for attempt in range(retries):
            with self._lock:
                url = self.urls[self._i % len(self.urls)]
                self._i += 1
            try:
                r = self._session().post(
                    f"{url}/v1/chat/completions", json=body, timeout=self.timeout
                )
                r.raise_for_status()
                d = r.json()
                usage = d["usage"]
                return {
                    "text": d["choices"][0]["message"]["content"],
                    "token_ids": usage["token_ids"],
                    "n_prompt": len(usage["prompt_token_ids"]),
                    "n_gen": len(usage["token_ids"]),
                }
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
            ) as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"generation failed after {retries} tries: {last}")


def render(
    tok, messages, tools, add_generation_prompt=True, enable_thinking=True
) -> str:
    return tok.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def parse_turn(text: str) -> dict:
    text = text.split("<|im_end|>")[0]
    m = THINK_RE.match(text)
    think = m.group(1).strip("\n") if m else ""
    body = text[m.end() :] if m else text
    calls, bad_json = ([], 0)
    for blob in CALL_RE.findall(body):
        try:
            obj = json.loads(blob)
            if not isinstance(obj, dict):
                raise TypeError("Tool call must be a JSON object")
        except (ValueError, TypeError):
            bad_json += 1
            continue
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = None
        calls.append({"name": obj.get("name"), "arguments": args})
    return {
        "text": text,
        "think": think,
        "content": CALL_RE.sub("", body).strip(),
        "calls": calls,
        "has_think": bool(m),
        "unclosed_think": "<think>" in text and "</think>" not in text,
        "nested_think": text.count("<think>") > 1,
        "unbalanced_think": text.count("<think>") != text.count("</think>"),
        "unclosed_call": text.count("<tool_call>") != text.count("</tool_call>"),
        "bad_call_json": bad_json,
    }


def to_message(parsed: dict) -> dict:
    msg = {"role": "assistant", "content": parsed["content"]}
    if parsed["think"]:
        msg["reasoning_content"] = parsed["think"]
    elif parsed.get("repaired_unclosed_think"):
        msg["reasoning_content"] = " "
    if parsed["calls"]:
        msg["tool_calls"] = [
            {
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for c in parsed["calls"]
        ]
    return msg


def canon(name, args) -> str:
    return json.dumps([name, args], sort_keys=True, ensure_ascii=False)


def call_key(call: dict) -> str:
    return canon(call["name"], call["arguments"])


def call_schema_error(call: dict, tools) -> str | None:
    name, args = (call.get("name"), call.get("arguments"))
    fn = next(
        (
            t.get("function") or {}
            for t in tools or []
            if (t.get("function") or {}).get("name") == name
        ),
        None,
    )
    if fn is None:
        return "unknown_tool"
    if not isinstance(args, dict):
        return "arguments_not_object"

    def value_error(value, schema, *, top=False):
        if not isinstance(schema, dict):
            return None
        typ = schema.get("type")
        if typ is None and (top or "properties" in schema or "required" in schema):
            typ = "object"
        if isinstance(typ, list):
            if not any(
                value_error(value, {**schema, "type": t}, top=top) is None for t in typ
            ):
                return "type"
            return None
        if typ == "object":
            if not isinstance(value, dict):
                return "type"
            props = schema.get("properties") or {}
            missing = set(schema.get("required") or []) - set(value)
            if missing:
                return "missing_required"
            if (top or schema.get("additionalProperties") is False) and set(
                value
            ) - set(props):
                return "extra_argument"
            for key, item in value.items():
                if key in props and value_error(item, props[key]) is not None:
                    return "nested_schema"
        elif typ == "array":
            if not isinstance(value, list):
                return "type"
            if any(
                value_error(item, schema.get("items") or {}) is not None
                for item in value
            ):
                return "nested_schema"
        elif (
            typ == "string"
            and (not isinstance(value, str))
            or typ == "integer"
            and (not isinstance(value, int) or isinstance(value, bool))
            or typ == "number"
            and (not isinstance(value, (int, float)) or isinstance(value, bool))
            or typ == "boolean"
            and (not isinstance(value, bool))
            or typ == "null"
            and value is not None
        ):
            return "type"
        if "enum" in schema and value not in schema["enum"]:
            return "enum"
        return None

    return value_error(args, fn.get("parameters") or {}, top=True)


def original_calls(
    messages: list[dict],
) -> tuple[list[dict[str, list[str]]], dict[str, str]]:
    episodes: list[dict[str, list[str]]] = []
    exact: dict[str, list[str]] | None = None
    by_name = {}
    for i, m in enumerate(messages):
        if m["role"] == "user":
            exact = {}
            episodes.append(exact)
            continue
        if exact is None:
            continue
        for j, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function", tc)
            resp = None
            k = i + 1 + j
            if k < len(messages) and messages[k]["role"] == "tool":
                resp = messages[k]["content"]
            if resp is None:
                continue
            exact.setdefault(
                canon(fn.get("name"), fn.get("arguments") or {}), []
            ).append(resp)
            by_name.setdefault(fn.get("name"), resp)
    return (episodes, by_name)


def episodes(messages: list[dict]) -> list[int]:
    out, cur, seen_user = ([], 0, False)
    for m in messages:
        if m["role"] == "user":
            if seen_user:
                out.append(cur)
            cur, seen_user = (0, True)
        elif m["role"] == "assistant" and seen_user:
            cur += 1
    if seen_user:
        out.append(cur)
    return out


class ContextOverflow(RuntimeError):
    pass


class Engine:
    def __init__(self, gen: Servers, sim: Servers | None, tok_gen, tok_sim, cfg):
        self.gen, self.sim, self.tok_gen, self.tok_sim, self.cfg = (
            gen,
            sim,
            tok_gen,
            tok_sim,
            cfg,
        )

    def turn(self, messages, tools) -> dict:
        prompt = render(self.tok_gen, messages, tools)
        n_prompt = len(self.tok_gen(prompt, add_special_tokens=False)["input_ids"])
        budget = MAX_SEQ_LEN - n_prompt
        if budget < 64:
            raise ContextOverflow(f"prompt {n_prompt} tokens leaves {budget}")
        out = self.gen.complete(
            prompt,
            min(self.cfg.max_new_tokens, budget),
            self.cfg.temperature,
            self.cfg.top_p,
            self.cfg.top_k,
        )
        p = parse_turn(out["text"])
        p["n_gen"] = out["n_gen"]
        p["n_prompt"] = out["n_prompt"]
        p["truncated"] = (
            bool(out["token_ids"]) and out["token_ids"][-1] != self.cfg.eos_id
        )
        if p["unclosed_think"] and (not p["truncated"]):
            p["content"] = p["content"].replace("<think>", "", 1).strip()
            p["repaired_unclosed_think"] = True
        return p

    def simulate(self, tools, call, user_ctx, reference) -> str:
        schema = next(
            (
                t
                for t in tools or []
                if (t.get("function") or {}).get("name") == call["name"]
            ),
            None,
        )
        blocks = [
            "### Function schema\n"
            + json.dumps(schema or {"name": call["name"]}, ensure_ascii=False),
            "### User request being served\n" + (user_ctx or "")[:600],
            "### Call\n" + json.dumps(call, ensure_ascii=False),
        ]
        if reference:
            try:
                ref = json.dumps(shape_of(json.loads(reference)), ensure_ascii=False)
            except (ValueError, TypeError):
                ref = ""
            if ref:
                blocks.append(
                    "### Response shape of this API — keys and value types only. Fill in values that fit THIS call\n"
                    + ref[:800]
                )
        prompt = render(
            self.tok_sim,
            [
                {"role": "system", "content": SIM_SYSTEM},
                {"role": "user", "content": "\n\n".join(blocks)},
            ],
            None,
            enable_thinking=False,
        )
        out = self.sim.complete(
            prompt, self.cfg.sim_max_tokens, self.cfg.sim_temperature
        )
        text = out["text"].split("<|im_end|>")[0].strip()
        m = FENCE_RE.match(text)
        if m:
            text = m.group(1).strip()
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True)
        except ValueError:
            return json.dumps(
                {"result": text[:1500]}, ensure_ascii=False, sort_keys=True
            )

    def run_sequential(self, trace: dict) -> dict:
        orig, tools = (trace["messages"], trace["tools"])
        exact_by_episode, by_name = original_calls(orig)
        users = [m for m in orig if m["role"] == "user"]
        budgets = episodes(orig)
        msgs: list[dict] = []
        if orig and orig[0]["role"] == "system":
            msgs.append(dict(orig[0]))
        turns, events, stop = ([], [], "complete")
        for ui, um in enumerate(users):
            exact = (
                copy.deepcopy(exact_by_episode[ui])
                if ui < len(exact_by_episode)
                else {}
            )
            msgs.append({"role": "user", "content": um["content"]})
            budget = min(
                max(budgets[ui] if ui < len(budgets) else 1, 1) + 2, self.cfg.max_steps
            )
            for step in range(budget):
                try:
                    p = self.turn(msgs, tools)
                except ContextOverflow:
                    stop = "overflow"
                    break
                turns.append({"episode": ui, "gen": p})
                msgs.append(to_message(p))
                if p["truncated"]:
                    stop = "truncated"
                    break
                if (
                    p["bad_call_json"]
                    or p["unclosed_call"]
                    or p["nested_think"]
                    or (
                        p["unbalanced_think"] and (not p.get("repaired_unclosed_think"))
                    )
                ):
                    stop = "malformed"
                    break
                schema_errors = [call_schema_error(c, tools) for c in p["calls"]]
                p["call_schema_errors"] = schema_errors
                if any(
                    e in ("unknown_tool", "arguments_not_object") for e in schema_errors
                ):
                    stop = "invalid_call"
                    break
                if not p["calls"]:
                    break
                if step == budget - 1:
                    stop = "budget"
                    break
                responses = []
                for c in p["calls"]:
                    k = call_key(c)
                    if exact.get(k):
                        responses.append((exact[k].pop(0), "replay"))
                    else:
                        responses.append(
                            (
                                self.simulate(
                                    tools, c, um["content"], by_name.get(c["name"])
                                ),
                                "sim",
                            )
                        )
                if stop != "complete":
                    break
                for r, src in responses:
                    msgs.append({"role": "tool", "content": r})
                    events.append(src)
            if stop != "complete":
                break
        return {"messages": msgs, "turns": turns, "events": events, "stop": stop}
