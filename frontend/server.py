import http.server
import socketserver
import os
import time
import subprocess
import json

PORT = 8080
# File paths
ROI_FILE = "/tmp/roi_config.txt"
TRIGGER_FILE = "/tmp/req_snapshot"
PREVIEW_FILE = "/tmp/preview.jpg"
RTSP_CONFIG_FILE = "rtsp_url.conf"
APP_LOG_FILE = "/tmp/app_debug.log"

# --- CONFIGURATION ---
# IMPORTANT: Use absolute path if possible
APP_DIR = "/opt/luckfox_pico_rtsp_yolov5_demo"
APP_BIN_NAME = "luckfox_pico_rtsp_yolov5"
APP_CMD = f"./{APP_BIN_NAME}"

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
            
        elif self.path == '/app_status':
            # Check if process is running using pidof
            is_running = os.system(f"pidof {APP_BIN_NAME} > /dev/null") == 0
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"running": is_running}).encode())

        elif self.path.startswith('/request_snapshot'):
            open(TRIGGER_FILE, 'a').close()
            for _ in range(20):
                if os.path.exists(PREVIEW_FILE) and os.path.getmtime(PREVIEW_FILE) > time.time() - 2:
                    break
                time.sleep(0.1)
            
            if os.path.exists(PREVIEW_FILE):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                with open(PREVIEW_FILE, 'rb') as f: self.wfile.write(f.read())
            else:
                self.send_error(404)

        elif self.path.startswith('/snapshot_image'):
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

        elif self.path == '/control_app':
            len = int(self.headers['Content-Length'])
            action = self.rfile.read(len).decode('utf-8').strip()
            self.send_response(200); self.end_headers()
            
            def manage_app():
                print(f"[Server] Action: {action}")
                
                if action in ['stop', 'restart']:
                    os.system(f"killall -9 {APP_BIN_NAME}")
                    time.sleep(1)

                if action in ['start', 'restart']:
                    try:
                        # CRITICAL FIX: Pass environment variables so libs load correctly
                        my_env = os.environ.copy()
                        # Add library paths if needed (usually handled by system, but good for debug)
                        # my_env["LD_LIBRARY_PATH"] = "/oem/usr/lib:/usr/lib" 

                        with open(APP_LOG_FILE, "a") as log:
                            log.write(f"\n--- Starting App at {time.ctime()} ---\n")
                            subprocess.Popen([APP_CMD], 
                                             cwd=APP_DIR,
                                             stdout=log, 
                                             stderr=log, 
                                             env=my_env, # <--- Fix: Pass Environment
                                             start_new_session=True)
                        print("[Server] App started")
                    except Exception as e:
                        print(f"[Server] Error starting app: {e}")

            import threading
            threading.Thread(target=manage_app).start()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Server started on port {PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass