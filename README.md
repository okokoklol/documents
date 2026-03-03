# OpenClaw-ish `AI` Documents Harness

`@ok.lol/documents` is a small library implementing an OpenClaw-inspired documents system that works in any runtime environment, for use with the Vercel AI SDK (v6).

Documents can be stored in an actual file system, SQLite, Postgres, etc. Implementing a `DocumentOps` subclass for your system is easy.

The API is tiny and composable.

## Usage

Use `getSystemPrompt` to assemble a system prompt with relevant documents, and `getTools` to give the agent document access.

The `messages` param is of type `ModelMessage[]`, so it works well with any `ai` construct, such as `ToolLoopAgent`.

```typescript
import BunSQLiteDocumentOps from "@ok.lol/documents/bun-sqlite";
import { Database } from "bun:sqlite";
import { ToolLoopAgent } from "ai";

// Build a DocumentOps instance backed by SQLite with in-memory search
const docs = new BunSQLiteDocumentOps({
  database: new Database(":memory:"),
  embeddingModel: "openai/text-embedding-3-small",
});

const agent = new ToolLoopAgent({
  model: "anthropic/claude-sonnet-4.6",
  tools: { ...docs.getTools(), otherAgentTools },

  // Inject relevant documents into system prompt on each call (generate, stream)
  prepareCall: async ({ messages }) => ({
    instructions: await docs.getSystemPrompt({
      instructions: "My base system prompt",
      messages,
    }),
  }),

  // Inject relevant documents into system prompt on each step (tool calls, thinking)
  prepareStep: async ({ messages }) => ({
    system: await docs.getSystemPrompt({
      instructions: "My base system prompt",
      messages,
    }),
  }),
});

// Use your agent
const result = agent.generate({ prompt: "What do I have coming up today?" });
```

## API

### `DocumentOps`

Base class for document operations. Extend this class, overriding all of its methods.

If your implementation has its own vector store (pgvector, Pinecone, etc.), extend `DocumentOps` directly and implement `search`. Otherwise, extend `DocumentOpsWithInMemorySearch` instead.

```typescript
import { DocumentOps } from "@ok.lol/documents";

class MyDocumentOps extends DocumentOps {
  async delete(params)  { /* ... */ }  // Deletes a document
  async edit(params)    { /* ... */ }  // Replaces a string in a document
  async glob(params)    { /* ... */ }  // Returns documents whose paths match a glob
  async grep(params)    { /* ... */ }  // Regular-expression-based content search
  async read(params)    { /* ... */ }  // Reads a document
  async search(params)  { /* ... */ }  // Searches documents by meaning (embeddings)
  async write(params)   { /* ... */ }  // Writes (creates or overwrites) a document
}
```

### `getTools()`

Returns a `Record<string, Tool>` the agent will use to access and manipulate documents.

```typescript
const docs = new MyDocumentOps();
const tools = docs.getTools();
// { delete, edit, glob, grep, read, search, write }, each AI SDK `tool`s
```

### `getSystemPrompt()`

Assembles a system prompt for the current conversation, injecting relevant documents into context.

```typescript
const systemPrompt = await docs.getSystemPrompt({
  instructions: "This is the base system prompt", // Base system prompt to extend with relevant documents
  alwaysInject: ["SOUL.MD"],                      // Glob patterns for documents to always inject before search-ranked results
  budget: 24_000,                                 // Maximum characters for the returned system prompt, including injected documents
  budgetPerDocument: 20_000,                      // Maximum characters to include per document before truncation, inclusive of the truncation message `[… ${truncatedChars} chars truncated]`
  messages: [],                                   // Conversation messages – type: ModelMessage[]
  neverInject: ["*.secret", "private/**"],        // Glob patterns for documents to never inject
});
```

### `DocumentOpsWithInMemorySearch`

Extends `DocumentOps` with an in-memory search implementation. Extend this class if you don't have a vector store.

This implementation lazily indexes all documents on first query, then keeps the index fresh on changes.

```typescript
import { DocumentOpsWithInMemorySearch } from "@ok.lol/documents";

// Extend DocumentOpsWithInMemorySearch — implement everything except search
class MyDocumentOps extends DocumentOpsWithInMemorySearch {
  constructor() {
    super("openai/text-embedding-3-small");
  }

  async delete(params)  { /* ... */ }
  async edit(params)    { /* ... call super.edit(params) after mutating ... */ }
  async glob(params)    { /* ... */ }
  async grep(params)    { /* ... */ }
  async read(params)    { /* ... */ }
  async write(params)   { /* ... call super.write(params) after mutating ... */ }
}
```

## Built-in DocumentOps

This package exports two built-in `DocumentOps` implementations.

**Bun SQLite**

DocumentOps backed by Bun SQLite.

```typescript
import BunSQLiteDocumentOps from "@ok.lol/documents/bun-sqlite";
import { Database } from "bun:sqlite";

// Pass in a Bun SQLite Database (this example uses an in-memory database)
const docs = new BunSQLiteDocumentOps({
  // This example uses an in-memory database.
  // You can pass a path instead to persist to disk.
  database: new Database(":memory:"),
  // The embedding model to use for in-memory search.
  embeddingModel: "openai/text-embedding-3-small",
});
```

**Bun Directory**

DocumentOps backed by Bun file APIs.

```typescript
import BunDirectoryDocumentOps from "@ok.lol/documents/bun-directory";
import { resolve } from "path";

const docs = new BunDirectoryDocumentOps({
  // Pass in the path to the root documents directory
  root: resolve("~/.myapp/documents"),
  // The embedding model to use for in-memory search.
  embeddingModel: "openai/text-embedding-3-small",
});
```
