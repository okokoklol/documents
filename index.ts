import { cosineSimilarity, embed, embedMany, tool, type EmbeddingModel, type ModelMessage } from "ai";
import z from "zod";

/**
 * Base class for document operations.
 * Extend this class to implement document operations, overriding all of its methods.
 * 
 * It requires `search` to be implemented, which is useful if you are using a vector
 * store like pgvector or pinecone. If your implementation doesn't have a vector store
 * (if backed by bun:sqlite or a directory, for example) extend DocumentOpsWithInMemorySearch
 * instead.
 */
export class DocumentOps {
  /** Deletes a document. */
  async delete(params: {
    /** File path. */
    path: string;
  }): Promise<void> {
    throw new DocumentOps.ERROR_MUST_BE_IMPLEMENTED('delete');
  }

  /** Replaces a unique string in a document. Fails if `old` is absent or ambiguous unless `all`. */
  async edit(params: {
    /** File path. */
    path: string;
    /** String to find. Must be unique in the file unless `all` is set. */
    old: string;
    /** Replacement string. */
    new: string;
    /** Replace every occurrence. Default: false. */
    all?: boolean;
  }): Promise<void> {
    throw new DocumentOps.ERROR_MUST_BE_IMPLEMENTED('edit');
  }

  /** Finds documents and directories by glob pattern. */
  async glob(params: {
    /** Glob pattern. Auto-prepended with `**\/` when not already recursive. */
    pattern: string;
    /** Root directory to search from. */
    path?: string;
    /** Cap results. */
    limit?: number;
  }): Promise<string[]> {
    throw new DocumentOps.ERROR_MUST_BE_IMPLEMENTED('glob');
  }

  /** Searches document contents by regex. */
  async grep(params: {
    /** Regex pattern. */
    pattern: string;
    /** Scope to a directory or file. */
    path?: string;
    /** Lines of context after each match. */
    after?: number;
    /** Lines of context before each match. */
    before?: number;
    /** Filter files by glob (e.g. `"*.ts"`). */
    fileGlob?: string;
    /** Case-insensitive matching. Default: false. */
    ignoreCase?: boolean;
    /** Cap results. */
    limit?: number;
    /** Enable multiline matching (`.` matches newlines). Default: false. */
    multiline?: boolean;
    /** Skip first N matches (pagination). */
    offset?: number;
  }): Promise<{
    /** Matched line content. */
    content: string;
    /** 1-based line number. */
    line: number;
    /** File path. */
    path: string;
  }[]> {
    throw new DocumentOps.ERROR_MUST_BE_IMPLEMENTED('grep');
  }

  /** Reads document contents, optionally a line range. */
  async read(params: {
    /** File path. */
    path: string;
    /** Max lines to return. */
    limit?: number;
    /** 1-based start line. */
    offset?: number;
  }): Promise<{
    /** File content (or requested line range). */
    content: string;
    /** Total lines in the file. */
    lines: number;
  }> {
    throw new DocumentOps.ERROR_MUST_BE_IMPLEMENTED('read');
  }

  /** Searches document contents by meaning (embeddings). */
  async search(params: {
    /** Natural-language query. */
    query: string;
    /** Cap results. */
    limit?: number;
    /** Scope to a directory or file. */
    path?: string;
  }): Promise<{
    /** Relevant text fragment. */
    content: string;
    /** File path. */
    path: string;
    /** Relevance, 0–1. */
    score: number;
  }[]> {
    throw new DocumentOps.ERROR_MUST_BE_IMPLEMENTED('search');
  }

  /** Creates or overwrites a document. */
  async write(params: {
    /** File path. */
    path: string;
    /** File content. */
    content: string;
  }): Promise<void> {
    throw new DocumentOps.ERROR_MUST_BE_IMPLEMENTED('write');
  }

  // –
  // System prompt
  // –

  /**
   * Assembles a system prompt by prepending optional instructions, then injecting
   * documents: first any always-inject paths, then search-ranked docs from the
   * last user/assistant messages. Respects a total character budget and per-document
   * truncation. Uses {@link read} and {@link search}.
   */
  async getSystemPrompt(params: {
    /** Base system prompt to prepend. */
    instructions?: string;
    /** Document paths (case-insensitive) to always inject before ranked results. */
    alwaysInject?: string[];
    /** Max characters for the entire returned prompt. Default: 24,000. */
    budget?: number;
    /** Max characters per document before truncation. Default: 20,000. */
    budgetPerDocument?: number;
    /** Conversation messages used to determine relevance for search. */
    messages: ModelMessage[];
  }): Promise<string> {
    // Total character budget for the assembled prompt
    const budget = params.budget ?? 24_000;
    // Max characters per document before truncation
    const perDoc = params.budgetPerDocument ?? 20_000;
    // Output segments (instructions, then optional <documents> block)
    const parts: string[] = [];
    // Characters left within the total budget after each step
    let remaining = budget;

    // Prepend optional base instructions
    if (params.instructions) {
      // Add instructions as first segment
      parts.push(params.instructions);
      // Consume budget
      remaining -= params.instructions.length;
    }

    // Documents to inject, in order: always-inject first, then search-ranked
    const docs: { content: string; path: string; score: number; truncated: boolean }[] = [];

    // Always-inject: read each path and add to docs until budget is exhausted
    if (params.alwaysInject?.length) {
      for (const path of params.alwaysInject) {
        // Stop when budget exhausted
        if (remaining <= 0) break;
        try {
          // Read document at path
          const { content } = await this.read({ path });
          // Truncate to fit per-doc and remaining budget
          const doc = systemPromptTruncate({ content, path, score: 1 }, Math.min(perDoc, remaining));
          docs.push(doc);
          // Consume budget
          remaining -= doc.content.length;
        } catch {
          // Path missing or unreadable; skip
        }
      }
    }

    // Search for relevant docs using text from the latest user + assistant messages
    // Natural-language query derived from the most recent user and assistant messages
    const query = systemPromptExtractQuery(params.messages);
    if (query && remaining > 0) {
      // Paths already in docs (always-inject), so we don't add them again from search
      const injectedPaths = new Set(docs.map((d) => d.path.toLowerCase()));
      // Search results deduped by path (first occurrence kept)
      const matches = systemPromptDedupe(await this.search({ query }));

      for (const match of matches) {
        if (remaining <= 0) break;
        if (injectedPaths.has(match.path.toLowerCase())) continue;
        // Read full content for this match
        const { content } = await this.read({ path: match.path });
        // Truncate to fit per-doc and remaining budget
        const doc = systemPromptTruncate(
          { content, path: match.path, score: match.score },
          Math.min(perDoc, remaining),
        );
        docs.push(doc);
        remaining -= doc.content.length;
      }
    }

    // Wrap docs in <documents> XML and append to parts
    if (docs.length) {
      // One <document> element per doc, with path, score, and optional truncated attr
      const items = docs.map(
        (d) =>
          `<document path="${d.path}" score="${d.score.toFixed(2)}"${d.truncated ? " truncated" : ""}>\n${d.content}\n</document>`,
      );
      // Append <documents> block to output
      parts.push(`<documents>\n${items.join("\n")}\n</documents>`);
    }

    // Join instructions and optional <documents> block with double newlines
    return parts.join("\n\n");
  }

  // –
  // AI SDK Tools Integration
  // –

  getTools() {
    const ops = this;

    return {
      /** Deletes a document. */
      delete: tool({
        description: "Deletes a document",
        inputSchema: z.object({
          path: z.string().describe("File path"),
        }),
        async execute(input) {
          await ops.delete(input);
          return { ok: true };
        },
      }),
      /** Replaces a unique string in a document. Fails if `old` is absent or ambiguous unless `all`. */
      edit: tool({
        description: "Replaces a unique string in a document. Fails if `old` is absent or ambiguous unless `all`",
        inputSchema: z.object({
          path: z.string().describe("File path"),
          old: z.string().describe("String to find (unique unless `all`)"),
          new: z.string().describe("Replacement"),
          all: z.boolean().optional().describe("Replace every occurrence"),
        }),
        async execute(input) {
          await ops.edit(input);
          return { ok: true };
        },
      }),
      /** Finds documents and directories by glob pattern. */
      glob: tool({
        description: "Finds documents and directories by glob pattern",
        inputSchema: z.object({
          pattern: z.string().describe("Glob pattern"),
          path: z.string().optional().describe("Root directory"),
          limit: z.number().optional().describe("Cap results"),
        }),
        async execute(input) {
          return ops.glob(input);
        },
      }),
      /** Searches document contents by regex. */
      grep: tool({
        description: "Searches document contents by regex",
        inputSchema: z.object({
          pattern: z.string().describe("Regex pattern"),
          path: z.string().optional().describe("Scope to directory or file"),
          after: z.number().optional().describe("Context lines after"),
          before: z.number().optional().describe("Context lines before"),
          fileGlob: z.string().optional().describe("Filter by glob"),
          ignoreCase: z.boolean().optional().describe("Case-insensitive"),
          limit: z.number().optional().describe("Cap results"),
        }),
        async execute(input) {
          return ops.grep(input);
        },
      }),
      /** Reads document contents, optionally a line range. */
      read: tool({
        description: "Reads document contents, optionally a line range",
        inputSchema: z.object({
          path: z.string().describe("File path"),
          limit: z.number().optional().describe("Max lines"),
          offset: z.number().optional().describe("1-based start line"),
        }),
        async execute(input) {
          return ops.read(input);
        },
      }),
      search: tool({
        description: "Searches document contents by meaning (embeddings)",
        inputSchema: z.object({
          query: z.string().describe("Natural-language query"),
          path: z.string().optional().describe("Scope to a directory or file"),
          limit: z.number().optional().describe("Cap results"),
        }),
        async execute(input) {
          return ops.search(input);
        },
      }),
      /** Creates or overwrites a document. */
      write: tool({
        description: "Creates or overwrites a document",
        inputSchema: z.object({
          path: z.string().describe("File path"),
          content: z.string().describe("File content"),
        }),
        async execute(input) {
          await ops.write(input);
          return { ok: true };
        },
      }),
    };
  }

  // –
  // Errors
  // –

  private static ERROR_MUST_BE_IMPLEMENTED = class ErrorMustBeImplemented extends Error {
    override name: string = "ErrorNotImplemented"
    constructor(method: string) { super(`${method} must be implemented by subclass.`) }
  }
}

/**
 * Base class for document operations, with an in-memory `search` implementation.
 * Extend this class to implement document operations, overriding all methods
 * except for `search`.
 * 
 * `search` will be implemented for you, which lazily indexes all documents on the 
 * first query, and then keeps the index fresh by intercepting mutations.
 */
export class DocumentOpsWithInMemorySearch extends DocumentOps {
  /** Whether the vectors index is initialized. */
  private initialized = false;
  /** In-memory index of vectors by document path. */
  private vectors = new Map<string, number[]>();

  /** Configure the embedding model for use in in-memory search. */
  constructor(
    /** Model to use for text embeddings. */
    private embeddingModel: EmbeddingModel,
    /** Max characters to embed per document. */
    private embeddingLimit = 8_000,
  ) { super() }

  /** Initializes the vectors index with any existing documents. */
  private async init() {
    // Bail if the vectors index is already initialized
    if (this.initialized) return;
    // Get all document paths
    const paths = await this.glob({ pattern: "**/*" });
    // Bail if there are no documents
    if (!paths.length) return;
    // Read the contents of all documents
    const readResults = await Promise.all(paths.map((p) => this.read({ path: p })));
    // Index all documents
    await this.index(paths.map((path, i) => ({ path, content: readResults[i]!.content })));
    // The index is now initialized
    this.initialized = true;
  }

  /** Indexes the content embedding for each provided {path, content} pair. */
  private async index(documents: { path: string; content: string }[]) {
    // Embed the content of each document, truncated at >`embeddingLimit`
    const { embeddings } = await embedMany({
      model: this.embeddingModel,
      values: documents.map(({ content }) => content.slice(0, this.embeddingLimit)),
    });
    // Index each path's embedding
    for (let i = 0; i < documents.length; i++) {
      this.vectors.set(documents[i]!.path, embeddings[i]!);
    }
  }

  /** Call super.delete(params) after performing the delete so the index stays in sync. */
  override async delete(params: { path: string; }): Promise<void> {
    this.vectors.delete(params.path);
  }

  /** Call super.edit(params) after performing the edit so the index stays in sync. */
  override async edit(params: { path: string; old: string; new: string; all?: boolean; }): Promise<void> {
    const { content } = await this.read({ path: params.path });
    await this.index([{ path: params.path, content }]);
  }

  /** Search document contents by meaning (embeddings). Uses in-memory search. */
  override async search(params: { query: string; limit?: number; path?: string; }): Promise<{ content: string; path: string; score: number; }[]> {
    await this.init();

    const { embedding: q } = await embed({
      model: this.embeddingModel,
      value: params.query,
    });

    let entries = [...this.vectors.entries()];
    if (params.path) {
      const prefix = params.path.endsWith("/")
        ? params.path
        : params.path + "/";
      entries = entries.filter(
        ([k]) => k === params.path || k.startsWith(prefix),
      );
    }

    const ranked = entries
      .map(([path, vec]) => ({ path, score: cosineSimilarity(q, vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, params.limit ?? 10);

    return Promise.all(
      ranked.map(async ({ path, score }) => ({
        content: (await this.read({ path })).content,
        path,
        score,
      })),
    );
  }

  /** Call super.write(params) after performing the write so the index stays in sync. */
  override async write(params: { path: string; content: string; }): Promise<void> {
    await this.index([params]);
  }
}

// –
// System prompt helpers
// –

/** Deduplicate by path, keeping first occurrence (assumes pre-sorted by score when used after search). */
function systemPromptDedupe<T extends { path: string }>(matches: T[]): T[] {
  // Paths we've already emitted
  const seen = new Set<string>();
  // Keep first occurrence per path
  return matches.filter((m) => {
    // Already emitted this path; drop
    if (seen.has(m.path)) return false;
    // Mark path as emitted
    seen.add(m.path);
    return true;
  });
}

/** Extracts searchable text from the most recent user message and the most recent assistant message (reverse order). */
function systemPromptExtractQuery(messages: ModelMessage[]): string | null {
  let userText: string | null = null;
  let assistantText: string | null = null;

  // Walk from latest message backwards to find most recent user and assistant
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!;
    if (msg.role === "user" && userText === null) {
      userText = systemPromptTextFromContent(msg.content).trim() || null;
    } else if (msg.role === "assistant" && assistantText === null) {
      assistantText = systemPromptTextFromContent(msg.content).trim() || null;
    }
    if (userText !== null && assistantText !== null) break;
  }

  // Concatenate user + assistant text for the search query
  const combined = [userText, assistantText].filter(Boolean).join(" ");
  return combined.trim() || null;
}

/** Flattens message content to a single string; supports string content or array of parts (text/reasoning). */
function systemPromptTextFromContent(content: string | Array<{ type?: string; text?: string }>): string {
  if (typeof content === "string") return content;
  // Content is array of parts
  // Keep only text/reasoning parts with string text
  return content
    .filter(
      (p): p is { type: "text" | "reasoning"; text: string } =>
        (p.type === "text" || p.type === "reasoning") && typeof p.text === "string",
    )
    // extract text from each part
    .map((p) => p.text)
    // single string
    .join(" ");
}

/** Truncate document content to `limit` chars and append a truncation suffix; set `truncated` on the result. */
function systemPromptTruncate(
  match: { content: string; path: string; score: number },
  limit: number,
): { content: string; path: string; score: number; truncated: boolean } {
  if (match.content.length <= limit) {
    // No truncation needed; pass through
    return { ...match, truncated: false };
  }
  // Number of characters we're cutting
  const truncatedChars = match.content.length - limit;
  // Suffix appended so caller knows content was truncated
  const suffix = `\n[… ${truncatedChars} chars truncated]`;
  // Slice to fit then append suffix
  const content = match.content.slice(0, Math.max(0, limit - suffix.length)) + suffix;
  // Return with truncated flag set
  return { content, path: match.path, score: match.score, truncated: true };
}