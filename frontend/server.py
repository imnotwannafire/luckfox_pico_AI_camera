import http.server
import socketserver
import os
import time

PORT = 8080
CONFIG_FILE = "/tmp/roi_config.txt"
TRIGGER_FILE = "/tmp/req_snapshot"
PREVIEW_FILE = "/tmp/preview.jpg"

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ROI Configurator</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #222; color: #eee; }
        #container { position: relative; display: inline-block; border: 1px solid #555; }
        img { display: block; max-width: 640px; }
        canvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
        button { padding: 10px 20px; margin: 10px; font-size: 16px; cursor: pointer; }
        .save { background-color: #4CAF50; color: white; border: none; }
        .refresh { background-color: #2196F3; color: white; border: none; }
        .clear { background-color: #f44336; color: white; border: none; }
    </style>
</head>
<body>
    <h2>ROI Configuration</h2>
    <div style="margin-bottom: 10px; color: #aaa; font-size: 0.9em;">
        1. Draw Polygon below &nbsp;&nbsp; 2. Click Save &nbsp;&nbsp; 3. Check RTSP Stream for result
    </div>
    
    <div id="container">
        <img id="snap" src="/snapshot_image" width="640" height="480">
        <canvas id="cvs" width="640" height="480"></canvas>
    </div>

    <div>
        <button class="refresh" onclick="location.reload()">Refresh Image</button>
        <button class="clear" onclick="clearPoly()">Clear</button>
        <button class="save" onclick="savePoly()">Save ROI</button>
    </div>
    <div id="msg"></div>

    <script>
        const cvs = document.getElementById('cvs');
        const ctx = cvs.getContext('2d');
        let pts = [];

        cvs.addEventListener('mousedown', e => {
            const r = cvs.getBoundingClientRect();
            pts.push({
                x: Math.round((e.clientX - r.left) * (cvs.width / r.width)),
                y: Math.round((e.clientY - r.top) * (cvs.height / r.height))
            });
            draw();
        });

        function draw() {
            ctx.clearRect(0, 0, cvs.width, cvs.height);
            if (!pts.length) return;
            
            ctx.strokeStyle = '#0f0'; ctx.lineWidth = 2; ctx.fillStyle = 'rgba(0,255,0,0.3)';
            ctx.beginPath();
            ctx.moveTo(pts[0].x, pts[0].y);
            pts.forEach(p => ctx.lineTo(p.x, p.y));
            if (pts.length > 2) ctx.closePath();
            ctx.stroke(); if (pts.length > 2) ctx.fill();
            
            ctx.fillStyle = '#f00';
            pts.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, 7); ctx.fill(); });
        }

        function clearPoly() { pts = []; draw(); }

        function savePoly() {
            if (pts.length < 3) return alert("Draw at least 3 points");
            const data = pts.map(p => `${p.x} ${p.y}`).join("\\n");
            fetch('/save', { method: 'POST', body: data }).then(() => {
                document.getElementById('msg').innerText = "Saved! Check RTSP stream.";
            });
        }
    </script>
</body>
</html>
"""

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Create trigger file to tell C++ to take a photo
            open(TRIGGER_FILE, 'a').close()
            # Wait briefly for C++ to write the file
            time.sleep(0.5) 
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
            
        elif self.path == '/snapshot_image':
            if os.path.exists(PREVIEW_FILE):
                with open(PREVIEW_FILE, 'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(f.read())
            else:
                self.send_error(404)

    def do_POST(self):
        if self.path == '/save':
            len = int(self.headers['Content-Length'])
            with open(CONFIG_FILE, 'wb') as f:
                f.write(self.rfile.read(len))
            self.send_response(200)
            self.end_headers()

if __name__ == "__main__":
    # Prevent socket bind errors
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", PORT), Handler)
    print(f"Config tool: http://<IP>:{PORT}")
    httpd.serve_forever()