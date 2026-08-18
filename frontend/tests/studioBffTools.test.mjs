import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = (path) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

const clientSource = source("../src/adk/client.ts");
const appSource = source("../src/App.tsx");
const composerSource = source("../src/ui/Composer.tsx");
const pickerSource = source("../src/ui/StudioToolPicker.tsx");
const stylesSource = source("../src/styles.css");

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
    "getRuntimeStudioToolCapabilities(\n      studioToolRuntime.runtimeId",
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

test("BFF tools can be selected before the first Session is created", () => {
  assert.match(
    composerSource,
    /const canOpenAddMenu = Boolean\(studioTools\) \|\| \(!disabled && allowAttachments\)/,
  );
  assert.match(composerSource, /disabled=\{!canOpenAddMenu\}/);
  assert.match(
    stylesSource,
    /\.composer--new-chat \.composer-menu-wrap\s*\{[\s\S]*?z-index:\s*3;/,
    "the new-chat textarea must not cover the add button",
  );
  assert.match(
    stylesSource,
    /\.welcome \.composer-slot\s*\{[\s\S]*?position:\s*relative;[\s\S]*?z-index:\s*20;/,
    "the tool picker must render above the welcome heading",
  );
  assert.match(appSource, /if \(!sessionId\) \{\s*setDraftStudioToolIds\(next\);\s*return;/);
  assert.match(
    appSource,
    /studioTools=\{[\s\S]*?studioToolRuntime &&[\s\S]*?newChatWorkspaceMode === "agent" &&[\s\S]*?newChatMode === "agent"/,
  );
  assert.match(
    appSource,
    /const studioToolRuntime = currentRuntime \?\? selectedDraftStudioRuntime/,
  );
  assert.match(
    appSource,
    /onConnected: \(agentId\) => \{[\s\S]*?setDraftStudioRuntime\(\{[\s\S]*?appName: agentId,[\s\S]*?runtimeId: runtime\.runtimeId,[\s\S]*?region: runtime\.region/,
  );
  assert.match(
    appSource,
    /getRuntimeStudioToolCapabilities\([\s\S]*?studioToolRuntime\.runtimeId,[\s\S]*?studioToolRuntime\.region/,
  );
  assert.doesNotMatch(
    appSource,
    /updateSelectedStudioToolIds[\s\S]{0,500}ensureSession/,
  );
});
