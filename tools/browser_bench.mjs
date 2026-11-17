#!/usr/bin/env node
// web/bench.html in headless Chrome, for browser timings that do not depend on a window being in view
// (a hidden or covered tab is throttled). Serve build/wasm/web first, then:
//
//   node tools/browser_bench.mjs 'http://localhost:8766/bench.html?mode=candidates&repeat=3' [more URLs]
//
// Prints one JSON line per URL: {url, status, log, rows}. Chrome runs with a fresh profile in a temporary
// directory and the HTTP cache off, so a rebuilt module is always the one measured. CHROME overrides the
// browser binary (default: Google Chrome on macOS); PORT the DevTools port (default 9333).
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const chrome = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = Number(process.env.PORT ?? 9333);
const profile = mkdtempSync(join(tmpdir(), "jpo-chrome-"));
const proc = spawn(chrome, ["--headless=new", "--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check",
                            `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`, "about:blank"],
                   { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const done = (code) => {
  proc.kill();
  setTimeout(() => {
    rmSync(profile, { recursive: true, force: true });
    process.exit(code);
  }, 500);
};

let version;
for (let i = 0; i < 100 && !version; ++i) {
  try {
    version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  } catch {
    await sleep(100);
  }
}
if (!version) {
  console.error("Chrome did not start: " + chrome);
  done(1);
} else {
  const ws = new WebSocket(version.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r));
  let next = 0;
  const pending = new Map();
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) {
      pending.get(m.id)(m);
      pending.delete(m.id);
    }
  });
  const send = (method, params = {}, sessionId) =>
    new Promise((r) => {
      const id = ++next;
      pending.set(id, r);
      ws.send(JSON.stringify({ id, method, params, sessionId }));
    });
  const { result: { targetId } } = await send("Target.createTarget", { url: "about:blank" });
  const { result: { sessionId } } = await send("Target.attachToTarget", { targetId, flatten: true });
  const evaluate = async (expression) =>
    (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, sessionId)).result?.result?.value;
  await send("Network.enable", {}, sessionId);
  await send("Network.setCacheDisabled", { cacheDisabled: true }, sessionId);
  for (const url of process.argv.slice(2)) {
    await send("Page.navigate", { url }, sessionId);
    await sleep(500);
    while (!(await evaluate("!!window.jpoBenchDone"))) await sleep(500);
    const out = await evaluate("JSON.stringify({status: document.getElementById('status').textContent, " +
                               "log: document.getElementById('log').textContent, rows: window.jpoBenchRows})");
    console.log(JSON.stringify({ url, ...JSON.parse(out) }));
  }
  done(0);
}
