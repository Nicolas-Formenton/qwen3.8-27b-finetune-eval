"""Prove the tau3 -> LiteLLM -> vLLM plumbing WITHOUT a GPU.

Starts a throwaway HTTP server that mimics vLLM's /v1/chat/completions, points
LiteLLM at it exactly the way run_tau3.sh does, and asserts that the request
arrived with the model name we expect. This is the same 'validate the instrument
offline' discipline used for BFCL: if this passes, the only unknown left in Phase 2
is the real model's behaviour, not our wiring.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8123
seen = {}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        seen["path"] = self.path
        seen["model"] = body.get("model")
        seen["messages"] = len(body.get("messages", []))
        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ROUTED_OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def main():
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    os.environ["OPENAI_API_BASE"] = f"http://127.0.0.1:{PORT}/v1"
    os.environ["OPENAI_API_KEY"] = "dummy"

    import litellm

    resp = litellm.completion(
        model="openai/qwen38-base",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
    )
    content = resp.choices[0].message.content
    srv.shutdown()

    print(f"path hit      : {seen.get('path')}")
    print(f"model seen    : {seen.get('model')}")
    print(f"content       : {content}")

    ok = (
        seen.get("path") == "/v1/chat/completions"
        and seen.get("model") == "qwen38-base"
        and content == "ROUTED_OK"
    )
    print("ROUTING VALIDATED" if ok else "ROUTING BROKEN")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
