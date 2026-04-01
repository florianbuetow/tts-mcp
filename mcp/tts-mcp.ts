/**
 * MCP server that relays TTS requests to the speech server.
 *
 * A transparent relay layer between MCP clients (Claude Code, Claude Desktop)
 * and the FastAPI TTS speech server. All responses are JSON. Errors from the
 * speech server are propagated as-is with full details — no retries, no
 * fallbacks, no swallowed errors.
 */

import { readFileSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = resolve(__dirname, "..");
const CONFIG_PATH = resolve(PROJECT_ROOT, "config.yaml");
const HEALTH_TIMEOUT_MS = 3_000;
const REQUEST_TIMEOUT_MS = 30_000;

type ToolResult = {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
};

function loadServerUrl(): string {
  let content: string;
  try {
    content = readFileSync(CONFIG_PATH, "utf-8");
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new Error(
      `Cannot read config file ${CONFIG_PATH}: ${detail}`,
    );
  }

  const hostMatch = content.match(/^host:\s*(.+)$/m);
  if (!hostMatch) {
    throw new Error(`Missing required key 'host' in ${CONFIG_PATH}`);
  }

  const portMatch = content.match(/^port:\s*(\d+)/m);
  if (!portMatch) {
    throw new Error(`Missing required key 'port' in ${CONFIG_PATH}`);
  }

  const host = hostMatch[1].trim();
  const port = portMatch[1].trim();

  // 0.0.0.0 is a listen address; connect via 127.0.0.1
  const connectHost = host === "0.0.0.0" ? "127.0.0.1" : host;

  return `http://${connectHost}:${port}`;
}

async function healthCheck(): Promise<ToolResult | null> {
  const baseUrl = loadServerUrl();
  const url = `${baseUrl}/health`;
  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    if (!response.ok) {
      const error = {
        error: "health_check_failed",
        url,
        message: `Speech server health check failed: HTTP ${response.status}`,
      };
      return {
        content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
        isError: true,
      };
    }
    const body = await response.json() as { status?: string };
    if (body?.status !== "ok") {
      const error = {
        error: "health_check_failed",
        url,
        message: "Speech server reported unhealthy status",
        details: body,
      };
      return {
        content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
        isError: true,
      };
    }
    return null;
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    const error = {
      error: "health_check_unreachable",
      url,
      message: `Speech server is not reachable at ${baseUrl}`,
      details: detail,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }
}

async function request(
  method: "GET" | "POST",
  path: string,
  body?: Record<string, unknown>,
): Promise<ToolResult> {
  const healthResult = await healthCheck();
  if (healthResult !== null) {
    return healthResult;
  }

  const baseUrl = loadServerUrl();
  const url = `${baseUrl}${path}`;

  let response: Response;
  try {
    const options: RequestInit = {
      method,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    };
    if (method === "POST" && body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    response = await fetch(url, options);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    const error = {
      error: "connection_failed",
      url,
      message: `Speech server is not reachable at ${baseUrl}`,
      details: detail,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }

  const responseText = await response.text();

  let responseBody: unknown;
  try {
    responseBody = JSON.parse(responseText);
  } catch {
    const error = {
      error: "invalid_json",
      status_code: response.status,
      url,
      raw_body: responseText.slice(0, 500),
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }

  if (!response.ok) {
    const error = {
      error: "http_error",
      status_code: response.status,
      url,
      response: responseBody,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }

  return {
    content: [{ type: "text", text: JSON.stringify(responseBody, null, 2) }],
  };
}

const server = new McpServer({
  name: "tts-mcp",
  version: "0.1.0",
});

server.tool(
  "say",
  "Queue text for speech synthesis and playback. Sends text to the TTS server which generates audio and plays it through speakers. Returns a message ID for status tracking.",
  {
    voice: z
      .string()
      .describe(
        "Voice to use for synthesis. Use get_voices to list available voices.",
      ),
    text: z.string().describe("Text to convert to speech."),
  },
  async ({ voice, text }) => {
    console.error(
      `[tts-mcp] say: voice=${voice} text="${text.slice(0, 80)}"`,
    );
    return request("POST", "/say", { text, voice });
  },
);

server.tool(
  "get_voices",
  "List all available TTS voices and the default voice from the speech server.",
  {},
  async () => {
    console.error("[tts-mcp] get_voices");
    return request("GET", "/voices");
  },
);

server.tool(
  "get_status",
  "Check status of a speech synthesis request. Returns status (queued/playing/completed/error), original text, audio file path, and error details.",
  {
    message_id: z
      .string()
      .describe("Message ID returned by the say tool."),
  },
  async ({ message_id }) => {
    console.error(`[tts-mcp] get_status: message_id=${message_id}`);
    return request(
      "GET",
      `/status/${encodeURIComponent(message_id)}`,
    );
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
