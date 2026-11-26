import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';

const port = 8781;
const pages = new Map([
  ['/', ['index.html', 'text/html']], ['/classifier', ['index.html', 'text/html']],
  ['/style.css', ['style.css', 'text/css']], ['/forms.js', ['forms.js', 'text/javascript']]
]);

async function handle(req, res) {
  function send(status, text, type = 'text/plain') {
    res.writeHead(status, {'Content-Type': type + '; charset=utf-8', 'Cache-Control': 'no-store'});
    res.end(text);
  }
  if (req.headers.host !== '127.0.0.1:' + port && req.headers.host !== 'localhost:' + port)
    return send(403, 'Local access only.');
  if (req.headers.origin && req.headers.origin !== 'http://' + req.headers.host)
    return send(403, 'Cross-origin requests are disabled.');
  const url = new URL(req.url, 'http://localhost');
  if (req.method === 'GET' && pages.has(url.pathname)) {
    const [file, type] = pages.get(url.pathname);
    return send(200, await readFile(new URL(file, import.meta.url)), type);
  }
  if (url.pathname !== '/api/predict' || req.method !== 'POST') return send(404, 'Page unavailable.');
  const target = 'http://127.0.0.1:8765/predict';
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 131072) return send(413, 'Input is too long.');
    chunks.push(chunk);
  }
  try {
    const upstream = await fetch(target, {
      method: req.method, body: Buffer.concat(chunks),
      headers: {'Content-Type': 'application/json'},
      signal: AbortSignal.timeout(12000)
    });
    res.writeHead(upstream.status, {'Content-Type': upstream.headers.get('content-type') || 'text/plain', 'Cache-Control':'no-store'});
    res.end(Buffer.from(await upstream.arrayBuffer()));
  } catch {
    send(503, 'Service unavailable. Check the project terminal and try again.');
  }
}
createServer((req, res) => handle(req, res).catch(() => {
  if (!res.headersSent) res.writeHead(500);
  res.end('Request failed.');
})).listen(port, '127.0.0.1', () => console.log('Frontend running at http://127.0.0.1:' + port));
