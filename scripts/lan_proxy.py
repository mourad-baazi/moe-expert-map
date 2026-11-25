"""Small relay so phones and other devices on the same network can use the llama-server chat page.

It forwards everything to the local llama-server (127.0.0.1:8080) but removes "tools" from chat requests,
because tool definitions trigger a llama.cpp bug with Inkling 2 Mini that breaks the server until restart.
Usage: py scripts/lan_proxy.py [--port 8090]      then open http://<PC address>:8090 on the phone.
Only devices on your local network can reach it if the firewall rule is limited to the local subnet.
"""
import argparse, http.client, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ("127.0.0.1", 8080)
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection", "content-length"}


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self):
        body = b""
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            body = self.rfile.read(n)
        if self.command == "POST" and "completions" in self.path and body:
            try:
                d = json.loads(body)
                for key in ("tools", "tool_choice"):
                    d.pop(key, None)
                body = json.dumps(d).encode()
            except ValueError:
                pass
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
        headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection(*UPSTREAM, timeout=900)
        conn.request(self.command, self.path, body=body or None, headers=headers)
        resp = conn.getresponse()
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        conn.close()
        self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = _forward

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    print(f"Relay on port {a.port} -> llama-server {UPSTREAM[0]}:{UPSTREAM[1]} (tools removed from requests)")
    ThreadingHTTPServer(("0.0.0.0", a.port), Relay).serve_forever()


if __name__ == "__main__":
    main()
