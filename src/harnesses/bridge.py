"""Local capability bridge: OpenAI chat inference and the four MCP environment tools.

The external harness is trusted controller code. Candidate Python still executes
only in the existing Docker sandbox. External trajectories never enter GRPO.
"""

import hmac
import json
import threading
import time
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from coding_rl.rollout.episode import bounded_observation
from environments.coding import tool_schemas
from harnesses.protocol import ProtocolError, ToolCall, dispatch, parse_response
from models.backend import GenerationTimeout
from utils.errors import InfrastructureError
from utils.seeding import derive_seed


def canonical_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI's JSON argument strings and tool IDs to template messages."""
    result, names = [], {}
    for original in messages:
        message = dict(original)
        if isinstance(message.get("content"), list):
            if any(part.get("type") != "text" for part in message["content"]):
                raise ValueError("The coding bridge accepts text messages only")
            message["content"] = "\n".join(p["text"] for p in message["content"])
        if message["role"] == "tool":
            message["name"] = message.get("name") or names.get(message.get("tool_call_id"))
            if not message["name"]:
                raise ValueError("Tool observation has no associated call")
        if message.get("tool_calls"):
            message["tool_calls"] = []
            for call in original["tool_calls"]:
                function = dict(call["function"])
                if isinstance(function["arguments"], str):
                    function["arguments"] = json.loads(function["arguments"])
                names[call.get("id")] = function["name"]
                message["tool_calls"].append({**call, "function": function})
        result.append(message)
    return result


class HarnessBridge:
    def __init__(self, environment, backend, task_id: str, attempt: int):
        self.environment, self.backend = environment, backend
        self.task_id, self.attempt = task_id, attempt
        self.token = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.deadline = time.monotonic() + environment.config.environment.episode_seconds
        self.turns = self.tokens = self.proposed = self.accepted = self.edits = self.rejected = (
            self.parser_errors
        ) = 0
        self.final = ""
        self.reason = "final_response"
        self.failure = None
        self.timings = {"generation_seconds": 0.0, "tool_seconds": 0.0, "peak_rss_bytes": 0}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def chat(self, request: dict) -> dict:
        settings = self.environment.config.rollout
        if self.turns >= settings.max_turns or self.tokens >= settings.episode_tokens:
            self.reason = "token_limit" if self.tokens >= settings.episode_tokens else "turn_limit"
            raise ValueError("Episode generation budget exhausted")
        messages = canonical_messages(request["messages"])
        ids, prefix = self.backend.render(messages, tools=request.get("tools", []))
        maximum = min(settings.max_new_tokens, settings.episode_tokens - self.tokens)
        if len(ids) + maximum + settings.observation_tokens + 64 > settings.context_tokens:
            self.reason = "context_limit"
            raise ValueError("External harness prompt exceeds the configured context reserve")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.reason = "episode_timeout"
            raise ValueError("Episode deadline exhausted")
        generated = self.backend.generate(
            ids,
            max_new_tokens=maximum,
            seed=derive_seed(
                self.environment.config.seed, self.task_id, self.attempt, self.turns + 1
            ),
            timeout=min(settings.generation_seconds, remaining),
            sampling=settings,
        )
        self.turns += 1
        self.tokens += len(generated.token_ids)
        self.timings["generation_seconds"] += generated.seconds
        self.timings["peak_rss_bytes"] = max(
            self.timings["peak_rss_bytes"], generated.peak_rss_bytes
        )
        self.environment.recorder.emit(
            "generation", input_ids=ids, generation_prefix=prefix, **asdict(generated)
        )
        message = {"role": "assistant", "content": generated.raw_text}
        finish = "stop"
        if generated.stop_reason != "eos":
            finish, self.reason = "length", "token_limit"
        else:
            try:
                parsed = parse_response(generated.raw_text, generation_prefix=prefix)
                message = parsed.message()
                self.final = parsed.content
                if parsed.call:
                    finish = "tool_calls"
                    message["tool_calls"][0]["id"] = "call_" + generated.request_id
                    message["tool_calls"][0]["function"]["arguments"] = json.dumps(
                        parsed.call.arguments
                    )
            except ProtocolError:
                self.parser_errors += 1
        return {
            "id": "chatcmpl-" + generated.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "policy",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(generated.token_ids),
                "total_tokens": len(ids) + len(generated.token_ids),
            },
        }

    def mcp(self, request: dict):
        method, params = request.get("method"), request.get("params", {})
        if method == "initialize":
            return {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "coding-rl", "version": "1.0"},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": name,
                        "description": schema["description"],
                        "inputSchema": schema["parameters"],
                    }
                    for name, schema in tool_schemas().items()
                ]
            }
        if method != "tools/call":
            raise ValueError("Unsupported MCP method")
        self.proposed += 1
        if self.proposed > self.environment.config.environment.max_tool_calls:
            self.reason = "tool_limit"
            result = {
                "ok": False,
                "error": "tool_limit",
                "message": "Episode tool budget exhausted",
            }
        else:
            before = time.monotonic()
            result = dispatch(
                self.environment, ToolCall(params["name"], params.get("arguments", {}), {})
            )
            self.timings["tool_seconds"] += time.monotonic() - before
        self.accepted += bool(result["ok"])
        self.edits += bool(result["ok"]) and params["name"] == "apply_patch"
        self.rejected += not result["ok"]
        observation = bounded_observation(
            result, self.backend, self.environment.config.rollout.observation_tokens
        )
        self.environment.recorder.emit(
            "model_observation", role="tool", name=params["name"], content=observation
        )
        return {"content": [{"type": "text", "text": observation}], "isError": not result["ok"]}

    def _handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                data = json.dumps(value, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                return hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + bridge.token
                )

            def do_GET(self):
                if not self.authorized():
                    return self.reply(401, {"error": "Unauthorized"})
                if self.path == "/v1/models":
                    return self.reply(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": "policy", "object": "model", "owned_by": "coding-rl"}],
                        },
                    )
                self.reply(405, {"error": "Server-initiated streams are unsupported"})

            def do_POST(self):
                if not self.authorized():
                    return self.reply(401, {"error": "Unauthorized"})
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2 * 1024 * 1024:
                    return self.reply(413, {"error": "Request exceeds bridge byte budget"})
                try:
                    request = json.loads(self.rfile.read(size))
                    with bridge.lock:
                        if time.monotonic() >= bridge.deadline:
                            bridge.reason = "episode_timeout"
                            raise ValueError("Episode deadline exhausted")
                        if self.path == "/mcp":
                            if "id" not in request:
                                return self.reply(202, {})
                            result = bridge.mcp(request)
                            return self.reply(
                                200, {"jsonrpc": "2.0", "id": request["id"], "result": result}
                            )
                        if self.path != "/v1/chat/completions":
                            return self.reply(404, {"error": "Unknown endpoint"})
                        result = bridge.chat(request)
                    if not request.get("stream"):
                        return self.reply(200, result)
                    choice = result["choices"][0]
                    delta = dict(choice["message"])
                    for i, call in enumerate(delta.get("tool_calls", [])):
                        call["index"] = i
                    chunk = {k: v for k, v in result.items() if k not in {"choices", "usage"}}
                    chunk["object"] = "chat.completion.chunk"
                    chunks = [
                        {**chunk, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                        {
                            **chunk,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
                            ],
                            "usage": result["usage"],
                        },
                    ]
                    data = (
                        "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                        + "data: [DONE]\n\n"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (InfrastructureError, GenerationTimeout) as exc:
                    bridge.failure = f"{type(exc).__name__}: {exc}"
                    self.reply(
                        503, {"error": {"message": bridge.failure, "type": "infrastructure_error"}}
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    self.reply(
                        400, {"error": {"message": str(exc), "type": "invalid_request_error"}}
                    )

        return Handler
