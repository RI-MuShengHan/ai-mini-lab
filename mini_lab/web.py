"""A local demo server. Run: python -m mini_lab.web --port 8765."""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from mini_lab import digits, wine
from mini_lab.search import NoteIndex

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def note_index(): return NoteIndex(ROOT / 'data/demo_notes')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # Don't log user-entered text.

    def send(self, status, value, content_type='application/json; charset=utf-8'):
        body = json.dumps(value, ensure_ascii=False).encode() if isinstance(value, (dict,list)) else value
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path)
        params = parse_qs(route.query)
        try:
            if route.path == '/':
                return self.send(200,(ROOT/'web/index.html').read_bytes(),'text/html; charset=utf-8')
            if route.path == '/api/digit':
                return self.send(200,digits.sample(int(params.get('index',['0'])[0])))
            if route.path == '/api/search':
                query = params.get('q',[''])[0]
                if len(query)>1000: raise ValueError('Query is too long.')
                return self.send(200,{'hits':note_index().search(query)})
            if route.path == '/api/wine':
                return self.send(200, wine.sample(int(params.get('index',['0'])[0])))
            if route.path == '/api/health':return self.send(200,{'status':'ok'})
            return self.send(404,{'error':'Not found'})
        except (ValueError,TypeError) as e:self.send(400,{'error':str(e)})
        except Exception:self.send(503,{'error':'Service unavailable. Run the evaluation CLI to diagnose model/data loading.'})

    def do_POST(self):
        if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            return self.send(403,{'error':'Cross-origin requests are not allowed.'})
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 65536:raise ValueError('Invalid request size.')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
            if self.path == '/api/predict-wine':
                return self.send(200,wine.predict_features(data.get('values',[])))
            if self.path == '/api/predict-digit':
                return self.send(200,{'predicted':digits.predict_pixels(data.get('pixels',[]))})
            self.send(404,{'error':'Not found'})
        except (ValueError,TypeError) as e:self.send(400,{'error':str(e)})
        except Exception:self.send(503,{'error':'Model unavailable. First run: python -m mini_lab.evaluate all'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Open http://127.0.0.1:{args.port} (Ctrl+C to stop)',flush=True)
    server.serve_forever()
