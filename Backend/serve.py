"""
SentinelAI — Local Development Server

Serves the frontend (HTML/CSS/JS) and the output videos
from the `outputs/` directory on http://localhost:8080.

Usage:
    python serve.py
"""

import http.server
import socketserver
import os
import sys
import urllib.parse
import mimetypes

PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


class SentinelHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that routes:
      /             → frontend/index.html
      /style.css    → frontend/style.css
      /script.js    → frontend/script.js
      /outputs/...  → outputs/<filename>
    """

    def do_GET(self):
        # Parse and decode the path
        parsed = urllib.parse.urlparse(self.path)
        decoded_path = urllib.parse.unquote(parsed.path)

        # Route: /outputs/<file>
        if decoded_path.startswith("/outputs/"):
            filename = decoded_path[len("/outputs/"):]
            filepath = os.path.join(OUTPUTS_DIR, filename)
            if os.path.isfile(filepath):
                self._serve_file_with_range(filepath)
                return
            else:
                self.send_error(404, f"Video not found: {filename}")
                return

        # Route: frontend static files
        if decoded_path == "/" or decoded_path == "":
            filepath = os.path.join(FRONTEND_DIR, "index.html")
        else:
            # Strip leading slash
            rel = decoded_path.lstrip("/")
            filepath = os.path.join(FRONTEND_DIR, rel)

        if os.path.isfile(filepath):
            self._serve_file_with_range(filepath)
        else:
            self.send_error(404, f"File not found: {decoded_path}")

    def _serve_file_with_range(self, filepath):
        """Serve a file with HTTP Range support (needed for video seeking)."""
        content_type, _ = mimetypes.guess_type(filepath)
        if content_type is None:
            content_type = "application/octet-stream"

        file_size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
            # Parse range: bytes=start-end
            try:
                range_val = range_header.strip().split("=")[1]
                parts = range_val.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else file_size - 1
            except (ValueError, IndexError):
                start = 0
                end = file_size - 1

            if start >= file_size:
                self.send_error(416, "Range Not Satisfiable")
                return

            end = min(end, file_size - 1)
            content_length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(filepath, "rb") as f:
                f.seek(start)
                remaining = content_length
                buf_size = 64 * 1024
                while remaining > 0:
                    chunk = f.read(min(buf_size, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            # Full file
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(filepath, "rb") as f:
                buf_size = 64 * 1024
                while True:
                    chunk = f.read(buf_size)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def log_message(self, format, *args):
        msg = format % args
        if "200" in msg or "206" in msg:
            print(f"  [OK] {msg}")
        elif "404" in msg:
            print(f"  [MISS] {msg}")
        else:
            print(f"  [LOG] {msg}")


def main():
    # Ensure directories exist
    if not os.path.isdir(FRONTEND_DIR):
        print(f"ERROR: Frontend directory not found: {FRONTEND_DIR}")
        sys.exit(1)
    if not os.path.isdir(OUTPUTS_DIR):
        print("WARNING: Outputs directory not found: " + OUTPUTS_DIR)
        print("         Videos will not be available.")

    # Register .mp4 MIME type explicitly
    mimetypes.add_type("video/mp4", ".mp4")

    # List available videos
    print("=" * 56)
    print("  SentinelAI -- Local Development Server")
    print("=" * 56)
    print("  Frontend: " + FRONTEND_DIR)
    print("  Videos:   " + OUTPUTS_DIR)
    print()

    if os.path.isdir(OUTPUTS_DIR):
        videos = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith(".mp4")]
        print(f"  Found {len(videos)} video(s):")
        for v in videos:
            size_mb = os.path.getsize(os.path.join(OUTPUTS_DIR, v)) / (1024 * 1024)
            print(f"     - {v}  ({size_mb:.1f} MB)")
    print()

    with socketserver.TCPServer(("", PORT), SentinelHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"  >> Server running at http://localhost:{PORT}")
        print("  Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


if __name__ == "__main__":
    main()
