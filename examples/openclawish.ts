/**
 * CLI chat with document-aware agent.
 *
 * Uses BunSQLiteDocumentOps for storage and ToolLoopAgent for agentic generation.
 * On first run, seeds the database with OpenClaw templates from GitHub.
 *
 * Env:
 *   ANTHROPIC_API_KEY / OPENAI_API_KEY — as needed by model providers
 *   MODEL       — laguage model (default: "anthropic/claude-sonnet-4.6" - via Vercel AI Gateway)
 *   EMBED_MODEL — embedding model (default: "openai/text-embedding-3-small" - via Vercel AI Gateway)
 *   DB_PATH     — sqlite file (default: documents.sqlite)
 *
 * Run:
 *   bun examples/cli.ts
 */

import { gateway, ToolLoopAgent, type ModelMessage } from "ai";
import { Database } from "bun:sqlite";
import * as readline from "readline";
import BunSQLiteDocumentOps from "../bun-sqlite";

// –
// Config
// –

const MODEL = process.env.MODEL ?? "anthropic/claude-sonnet-4.6";
const EMBED_MODEL = process.env.EMBED_MODEL ?? "openai/text-embedding-3-small";
const DB_PATH = process.env.DB_PATH ?? "documents.sqlite";

// –
// Documents
// –

const docs = new BunSQLiteDocumentOps({
  database: new Database(DB_PATH),
  embeddingModel: gateway.embeddingModel(EMBED_MODEL),
});

const existing = await docs.glob({ pattern: "**/*" });
if (!existing.length) {
  process.stderr.write("Seeding documents from GitHub...\n");
  await seed(docs);
  const paths = await docs.glob({ pattern: "**/*" });
  process.stderr.write(`Seeded ${paths.length} documents.\n`);
}

// –
// Agent
// –

const instructions = "You are a helpful assistant with access to a documents store.";

const agent = new ToolLoopAgent({
  model: gateway(MODEL),
  tools: docs.getTools(),

  // Inject relevant documents into the system prompt
  async prepareCall(params) {
    return {
      ...params,
      instructions: await docs.getSystemPrompt({ instructions, messages: params.messages ?? [] }),
    };
  },

  // Re-inject after tool use in case documents changed
  async prepareStep({ stepNumber, messages }) {
    if (stepNumber === 0) return;
    return {
      system: await docs.getSystemPrompt({ instructions, messages }),
    };
  },
});

// –
// REPL
// –

const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const ask = () => new Promise<string>((resolve) => rl.question("> ", resolve));
const messages: ModelMessage[] = [];

process.stderr.write(`${MODEL} | ${EMBED_MODEL} | ${DB_PATH}\n\n`);

while (true) {
  const input = await ask();
  if (!input.trim()) continue;
  if (input.trim() === "exit") break;

  messages.push({ role: "user", content: input });

  try {
    const result = await agent.stream({ messages });

    for await (const part of result.fullStream) {
      switch (part.type) {
        case "text-delta":
          process.stdout.write(part.text);
          break;
        case "tool-call": {
          const json = JSON.stringify(part.input);
          process.stdout.write(dim(`\n[${part.toolName}(${json})]\n`));
          break;
        }
        case "tool-result": {
          const json = JSON.stringify(part.output);
          const preview = json.length > 200 ? json.slice(0, 200) + "..." : json;
          process.stdout.write(dim(`-> ${preview}\n`));
          break;
        }
      }
    }

    process.stdout.write("\n\n");

    const { messages: responseMessages } = await result.response;
    messages.push(...responseMessages);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(`\x1b[31mError: ${msg}\x1b[0m\n\n`);
  }
}

rl.close();

// –
// Seed
// –

/** Fetches OpenClaw templates from GitHub and writes them into the document store. */
async function seed(docs: BunSQLiteDocumentOps): Promise<void> {
  const repo = "openclaw/openclaw";
  const dir = "docs/reference/templates";

  const res = await fetch(`https://api.github.com/repos/${repo}/contents/${dir}?ref=main`);
  if (!res.ok) throw new Error(`GitHub API ${res.status}: ${res.statusText}`);

  const entries = await res.json();
  if (!Array.isArray(entries)) throw new Error("Unexpected GitHub API response");

  await Promise.all(
    entries
      .filter((e): e is { name: string; download_url: string } =>
        typeof e?.name === "string" && typeof e?.download_url === "string")
      .map(async ({ name, download_url }) => {
        const file = await fetch(download_url);
        if (!file.ok) throw new Error(`Failed to fetch ${name}: ${file.status}`);
        await docs.write({ path: name, content: await file.text() });
      }),
  );
}
