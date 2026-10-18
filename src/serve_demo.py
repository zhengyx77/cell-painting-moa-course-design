"""Local-only demo and actual model inference. No remote service or uploads."""
from pathlib import Path
import sys, json, io, argparse, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.packages').exists(): sys.path.insert(0,str(ROOT/'.packages'))
from predict import predict_frame
import pandas as pd

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/api/health':
            ready=(ROOT/'models/selected_model.joblib').exists()
            b=json.dumps({'status':'ready' if ready else 'missing_model','model':'RF_leaf1','local_only':True}).encode()
            self.send_response(200 if ready else 503);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
            return
        super().do_GET()
    def do_POST(self):
        if self.path!='/api/predict': self.send_error(404);return
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<1 or n>20_000_000: raise ValueError('File size must be between 1 byte and 20 MB')
            raw=self.rfile.read(n).decode('utf-8-sig')
            result=predict_frame(pd.read_csv(io.StringIO(raw)))
            payload={'predictions':result.to_dict('records')}
            code=200
        except Exception as e: payload={'error':str(e)};code=400
        b=json.dumps(payload,ensure_ascii=False).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--no-browser',action='store_true');a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(ROOT)))
    url=f'http://127.0.0.1:{a.port}/demo/index.html'
    print('Demo:',url,flush=True)
    if not a.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
