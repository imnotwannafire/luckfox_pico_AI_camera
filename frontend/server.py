import http.server
import socketserver
import os
import time
import subprocess

PORT = 8080
# File paths
ROI_FILE = "/tmp/roi_config.txt"
TRIGGER_FILE = "/tmp/req_snapshot"
PREVIEW_FILE = "/tmp/preview.jpg"
RTSP_CONFIG_FILE = "rtsp_url.conf"

# App Path (Relative to server.py)
APP_DIR = "../luckfox_pico_rtsp_yolov5_demo"
APP_BIN = "./luckfox_pico_rtsp_yolov5"

# Ensure we can find index.html regardless of where python is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            if not os.path.exists(INDEX_FILE):
                self.send_error(404, "index.html not found.")
                return
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open(INDEX_FILE, 'rb') as f: self.wfile.write(f.read())
            
        elif self.path.startswith('/request_snapshot'):
            # Create trigger file
            open(TRIGGER_FILE, 'a').close()
            # Wait loop
            found = False
            for _ in range(20): # 2 seconds max
                if os.path.exists(PREVIEW_FILE) and os.path.getmtime(PREVIEW_FILE) > time.time() - 2:
                    found = True
                    break
                time.sleep(0.1)
            
            if os.path.exists(PREVIEW_FILE):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                with open(PREVIEW_FILE, 'rb') as f: self.wfile.write(f.read())
            else:
                self.send_error(404)

        elif self.path.startswith('/snapshot_image'):
            # Fallback direct access
            if os.path.exists(PREVIEW_FILE):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                with open(PREVIEW_FILE, 'rb') as f: self.wfile.write(f.read())
            else:
                self.send_error(404)

        elif self.path == '/get_rtsp':
            self.send_response(200)
            self.end_headers()
            if os.path.exists(RTSP_CONFIG_FILE):
                with open(RTSP_CONFIG_FILE, 'rb') as f: self.wfile.write(f.read().strip())
            else:
                self.wfile.write(b"")

    def do_POST(self):
        if self.path == '/save_roi':
            len = int(self.headers['Content-Length'])
            with open(ROI_FILE, 'wb') as f: f.write(self.rfile.read(len))
            self.send_response(200); self.end_headers()

        elif self.path == '/save_rtsp':
            len = int(self.headers['Content-Length'])
            with open(RTSP_CONFIG_FILE, 'wb') as f: f.write(self.rfile.read(len))
            self.send_response(200); self.end_headers()

        elif self.path == '/restart_app':
            self.send_response(200); self.end_headers()
            
            def restart_logic():
                print("[Server] Restarting Main Application...")
                # 1. Kill (using binary name)
                os.system(f"killall -9 {os.path.basename(APP_BIN)}")
                time.sleep(1)
                
                # 2. Start
                try:
                    # We switch CWD so the app finds its models/libs correctly
                    subprocess.Popen([APP_BIN], 
                                     cwd=APP_DIR,
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL, 
                                     start_new_session=True)
                    print("[Server] App launched successfully")
                except Exception as e:
                    print(f"[Server] Failed to launch app: {e}")

            import threading
            threading.Thread(target=restart_logic).start()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving UI at http://0.0.0.0:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
