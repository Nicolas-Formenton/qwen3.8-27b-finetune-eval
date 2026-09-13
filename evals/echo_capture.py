"""Tiny OpenAI-compatible echo server: captures request params (esp. max_tokens) to a JSONL file.

Used to prove what the Hermes client actually sends for a custom provider. Speaks SSE so the
agent's streaming path is exercised exactly as with the real vLLM endpoint.
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAPTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured.jsonl")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):  # /v1/models
        body = json.dumps({"object": "list", "data": [{"id": "echo-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw)
        except Exception:
            req = {"_unparsed": raw[:500].decode("utf-8", "replace")}

        rec = {
            "ts": time.strftime("%H:%M:%S"),
            "path": self.path,
            "max_tokens": req.get("max_tokens"),
            "max_completion_tokens": req.get("max_completion_tokens"),
            "model": req.get("model"),
            "stream": req.get("stream"),
            "n_messages": len(req.get("messages") or []),
            "n_tools": len(req.get("tools") or []),
            "extra_body_keys": sorted(k for k in req if k not in (
                "model", "messages", "tools", "stream", "max_tokens", "max_completion_tokens")),
            "sampling": {k: req.get(k) for k in ("temperature", "top_p", "seed", "reasoning_effort")},
        }
        with open(CAPTURE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(delta, finish=None):
                payload = {
                    "id": "chatcmpl-echo", "object": "chat.completion.chunk", "created": 0,
                    "model": "echo-model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                data = f"data: {json.dumps(payload)}\n\n".encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            chunk({"role": "assistant", "content": "ok"})
            chunk({}, "stop")
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            body = json.dumps({
                "id": "chatcmpl-echo", "object": "chat.completion", "created": 0, "model": "echo-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    open(CAPTURE, "w").close()
    print("echo server on 8099, capture ->", CAPTURE, flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8099), H).serve_forever()
