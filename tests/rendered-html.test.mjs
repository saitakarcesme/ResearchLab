import assert from "node:assert/strict";
import test from "node:test";

async function render(pathname) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${pathname}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

for (const [pathname, title, visibleCopy] of [
  ["/lab", "Lab", "What do you want to research?"],
  ["/researches", "Researches", "Researches"],
  ["/articles", "Articles", "Articles"],
]) {
  test(`server-renders ${pathname}`, async () => {
    const response = await render(pathname);
    assert.equal(response.status, 200);
    assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
    const html = await response.text();
    assert.match(html, new RegExp(`<title>${title}[^<]*Autoresearch Lab<\\/title>`, "i"));
    assert.match(html, new RegExp(visibleCopy.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"));
    assert.match(html, />Lab</);
    assert.match(html, />Researches</);
    assert.match(html, />Articles</);
    assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
  });
}
