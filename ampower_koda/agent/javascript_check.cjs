"use strict";

// Inspect source without executing it or loading executable project configs.
// Keep this check narrow: formatting and unused variables do not block repairs.
const fs = require("node:fs");
const path = require("node:path");

// Lowercase browser globals allowed bare. Other window properties (name, status, top...)
// would hide an out-of-scope local, so they must be written window.name.
const BARE_BROWSER = new Set([
  "window", "document", "navigator", "location", "history", "console", "localStorage",
  "sessionStorage", "indexedDB", "caches", "crypto", "performance", "customElements", "fetch",
  "alert", "confirm", "prompt", "setTimeout", "clearTimeout", "setInterval", "clearInterval",
  "requestAnimationFrame", "cancelAnimationFrame", "requestIdleCallback", "cancelIdleCallback",
  "queueMicrotask", "structuredClone", "atob", "btoa", "getComputedStyle", "getSelection",
  "matchMedia", "createImageBitmap", "postMessage", "reportError", "addEventListener",
  "removeEventListener", "dispatchEvent", "devicePixelRatio", "innerWidth", "innerHeight",
  "scrollX", "scrollY", "pageXOffset", "pageYOffset", "visualViewport", "isSecureContext",
]);

function browserGlobals(all) {
  // Constructors and interfaces (URL, ResizeObserver, HTMLElement) stay; they are not local names.
  return Object.fromEntries(Object.entries(all).filter(([name]) => !/^[a-z]/.test(name) || BARE_BROWSER.has(name)));
}

function check(input) {
  const { Linter } = require("eslint");
  const globals = require("globals");
  const jsonc = require("jsonc-parser");
  if (Number(Linter.version.split(".")[0]) !== 9) {
    throw new Error("Install the agent's declared ESLint 9 dependency with npm ci.");
  }
  const envs = { ...globals, browser: browserGlobals(globals.browser) };
  const known = { ...envs.browser, ...globals.node, frappe: "readonly", __: "readonly" };
  const configs = [];
  for (const filename of input.configs || []) {
    if (!fs.existsSync(filename)) continue;
    const errors = [];
    const parsed = jsonc.parse(fs.readFileSync(filename, "utf8"), errors, { allowTrailingComma: true });
    if (errors.length || !parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`Cannot read JavaScript globals from ${filename}: expected a JSON/JSONC object.`);
    }
    const config = path.basename(filename) === "package.json" ? parsed.eslintConfig : parsed;
    if (!config) continue;
    for (const [env, enabled] of Object.entries(config.env || {})) {
      if (enabled && envs[env]) Object.assign(known, envs[env]);
    }
    Object.assign(known, config.globals || {});
    configs.push(filename);
  }
  const messages = new Linter().verify(input.source, {
    languageOptions: { ecmaVersion: "latest", sourceType: "module", globals: known },
    // A source edit must not turn off the check that is checking that edit.
    linterOptions: { noInlineConfig: true },
    rules: { "no-undef": "error" },
  }, { filename: "source.js" }).filter(message => message.severity === 2);
  return {
    checked: true,
    count: messages.length,
    diagnostics: messages.slice(0, 20).map(({ ruleId, line, column, message }) => ({ ruleId, line, column, message })),
    configs,
  };
}

try {
  const input = JSON.parse(fs.readFileSync(0, "utf8"));
  process.stdout.write(JSON.stringify(check(input)));
} catch (error) {
  process.stdout.write(JSON.stringify({ checked: false, error: String(error.message).slice(0, 1000) }));
  process.exitCode = 2;
}
