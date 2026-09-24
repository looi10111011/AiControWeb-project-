const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const source = readFileSync(require('node:path').join(__dirname, '../app/static/ai-bar.js'), 'utf8');

function mount(storage = new Map()) {
  const elements = new Map();
  const streams = [];
  const requests = [];
  let uuid = 0;
  const makeClassList = () => {
    const classes = new Set();
    return {
      add: name => classes.add(name),
      remove: name => classes.delete(name),
      contains: name => classes.has(name),
      toggle: (name, force = !classes.has(name)) => (force ? classes.add(name) : classes.delete(name), force),
    };
  };
  const makeElement = () => ({
    dataset: {}, style: {}, value: '', disabled: false, hidden: false, handlers: {}, scrollHeight: 0,
    classList: makeClassList(), appendChild() {},
    addEventListener(name, handler) { this.handlers[name] = handler; },
    attachShadow() { return this; },
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, makeElement());
      return elements.get(selector);
    },
  });
  const slot = makeElement();
  const context = {
    setTimeout() {},
    crypto: { randomUUID: () => `uuid-${++uuid}` },
    sessionStorage: {
      getItem: key => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    MutationObserver: class { observe() {} },
    document: {
      createElement: makeElement, body: makeElement(), documentElement: makeElement(),
      getElementById: id => (id === 'hermes-ai-bar-slot' ? slot : null),
    },
    window: {
      HermesPageBridge: { documentId: 'document-1', snapshot: () => ({ document_id: 'document-1', url: 'http://localhost:8100/dashboard', elements: [] }), execute: async () => 'OK' },
      location: { protocol: 'http:', hostname: 'localhost', href: 'http://localhost:8100/dashboard' },
      addEventListener() {},
    },
    fetch: async (url, options) => {
      if (url.endsWith('/page-bridge')) return { ok: true, json: async () => ({ execution: 'in_page' }) };
      if (url.endsWith('/page')) return { ok: true, json: async () => ({ command: null }) };
      requests.push({ url, body: JSON.parse(options.body) });
      return { ok: true, json: async () => ({ task_id: 'task-1', embedded_token: 'bridge-token' }) };
    },
    EventSource: class {
      constructor(url) { this.url = url; streams.push(this); }
      close() { this.closed = true; }
    },
  };
  vm.runInNewContext(source, context);
  return { elements, streams, requests, storage };
}

test('send binds the current document and rejects duplicate submission', async () => {
  const app = mount();
  app.elements.get('.bar textarea').value = 'ไปหน้า PIM';
  await app.elements.get('.send-btn').handlers.click();
  await app.elements.get('.send-btn').handlers.click();
  assert.equal(app.requests.length, 1);
  assert.equal(app.elements.get('.bar').classList.contains('thinking'), true);
  assert.equal(app.requests[0].body.embedded_page.document_id, 'document-1');
  assert.equal(app.requests[0].body.use_user_browser, undefined);
  assert.equal(app.requests[0].body.url, 'http://localhost:8100/dashboard');
  assert.equal(app.storage.get('hermesAiBarTaskId'), 'task-1');
});

test('navigation restores the task stream and completion unlocks the bar', () => {
  const storage = new Map([['hermesAiBarTaskId', 'task-1'], ['hermesAiBarBridgeToken', 'bridge-token']]);
  const app = mount(storage);
  assert.equal(app.elements.get('.send-btn').disabled, true);
  assert.equal(app.streams[0].url, 'http://localhost:8000/tasks/task-1/stream');
  app.streams[0].onmessage({ data: JSON.stringify({ kind: 'task_done', status: 'done' }) });
  assert.equal(app.elements.get('.send-btn').disabled, false);
  assert.equal(app.elements.get('.bar').classList.contains('thinking'), false);
  assert.equal(storage.has('hermesAiBarTaskId'), false);
  assert.equal(app.streams[0].closed, true);
});
