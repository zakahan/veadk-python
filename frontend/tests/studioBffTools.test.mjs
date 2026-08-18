import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = (path) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

const clientSource = source("../src/adk/client.ts");
const appSource = source("../src/App.tsx");
const composerSource = source("../src/ui/Composer.tsx");
const pickerSource = source("../src/ui/StudioToolPicker.tsx");

test("runSSE sends an explicit per-run platform tool selection", () => {
  assert.match(clientSource, /platformTools\?: readonly string\[\]/);
  assert.match(clientSource, /platform_tools: \[\.\.\.platformTools\]/);
  assert.match(
    clientSource,
    /runtime-tool-channel\/\$\{encodeURIComponent\(runtimeId\)\}\/capabilities/,
  );
});

test("Studio keeps BFF tool selection separate per session", () => {
  assert.match(appSource, /studioToolIdsBySession/);
  assert.match(appSource, /studioToolSelectionKey\(appName, userId, sessionId\)/);
  assert.match(appSource, /platformTools: currentRuntime \? platformTools : undefined/);
  assert.match(appSource, /selectedIds: selectedStudioToolIds/);
});

test("BFF tool discovery keeps a stable hook order across login", () => {
  const capabilityCall = appSource.indexOf(
    "getRuntimeStudioToolCapabilities(\n      currentRuntime.runtimeId",
  );
  const authenticationReturn = appSource.indexOf("if (authError) {");

  assert.ok(capabilityCall >= 0, "capability discovery should be present");
  assert.ok(authenticationReturn >= 0, "authentication gate should be present");
  assert.ok(
    capabilityCall < authenticationReturn,
    "capability hook must execute before conditional authentication returns",
  );
});

test("Composer exposes accessible local tool switches and selected chips", () => {
  assert.match(composerSource, /<StudioToolPicker/);
  assert.match(composerSource, /<StudioToolChips/);
  assert.match(pickerSource, /role="switch"/);
  assert.match(pickerSource, /aria-checked=\{checked\}/);
  assert.match(pickerSource, /结果会发送给云端 Agent/);
  assert.doesNotMatch(pickerSource, /from "lucide-react"/);
});
