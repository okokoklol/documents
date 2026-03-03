import { describe, expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { MockEmbeddingModelV3 } from "ai/test";
import { DocumentOps, DocumentOpsWithInMemorySearch } from "./index";

// –
// Helpers
// –

/** Deterministic embedding from text content. */
function hashToVector(text: string, dim = 16): number[] {
  const vec = new Array(dim).fill(0);
  for (let i = 0; i < text.length; i++) vec[i % dim] += text.charCodeAt(i);
  const norm = Math.sqrt(vec.reduce((s: number, v: number) => s + v * v, 0)) || 1;
  return vec.map((v: number) => v / norm);
}

function mockEmbeddingModel() {
  return new MockEmbeddingModelV3({
    doEmbed: async ({ values }) => ({
      embeddings: values.map((v) => hashToVector(v)),
      warnings: [],
    }),
  });
}

/** Minimal DocumentOps with controllable read/glob/search for testing getSystemPrompt. */
class StubDocumentOps extends DocumentOps {
  constructor(
    private docs: Map<string, string>,
    private searchResults: { content: string; path: string; score: number }[] = [],
  ) {
    super();
  }

  override async glob({ pattern }: { pattern: string }) {
    const effective = pattern.startsWith("**/") ? pattern : `**/${pattern}`;
    const g = new Bun.Glob(effective);
    return [...this.docs.keys()].filter((p) => g.match(p));
  }

  override async read({ path }: { path: string }) {
    const content = this.docs.get(path);
    if (!content) throw new Error("not found");
    return { content, lines: content.split("\n").length };
  }

  override async search() {
    return this.searchResults;
  }
}

/** Map-backed DocumentOpsWithInMemorySearch for testing search behavior. */
class MapDocumentOps extends DocumentOpsWithInMemorySearch {
  private store = new Map<string, string>();

  override async delete(params: { path: string }) {
    this.store.delete(params.path);
    await super.delete(params);
  }

  override async edit(params: { path: string; old: string; new: string; all?: boolean }) {
    const content = this.store.get(params.path);
    if (!content) throw new Error("not found");
    this.store.set(
      params.path,
      params.all ? content.replaceAll(params.old, params.new) : content.replace(params.old, params.new),
    );
    await super.edit(params);
  }

  override async glob() {
    return [...this.store.keys()];
  }

  override async grep() {
    return [];
  }

  override async read({ path }: { path: string }) {
    const content = this.store.get(path);
    if (content === undefined) throw new Error("not found");
    return { content, lines: content.split("\n").length };
  }

  override async write(params: { path: string; content: string }) {
    this.store.set(params.path, params.content);
    await super.write(params);
  }
}

// –
// DocumentOps
// –

describe("DocumentOps", () => {
  const ops = new DocumentOps();

  test("delete throws", () => expect(ops.delete({ path: "x" })).rejects.toThrow());
  test("edit throws", () => expect(ops.edit({ path: "x", old: "a", new: "b" })).rejects.toThrow());
  test("glob throws", () => expect(ops.glob({ pattern: "*" })).rejects.toThrow());
  test("grep throws", () => expect(ops.grep({ pattern: "x" })).rejects.toThrow());
  test("read throws", () => expect(ops.read({ path: "x" })).rejects.toThrow());
  test("search throws", () => expect(ops.search({ query: "x" })).rejects.toThrow());
  test("write throws", () => expect(ops.write({ path: "x", content: "" })).rejects.toThrow());
});

// –
// DocumentOpsWithInMemorySearch
// –

describe("DocumentOpsWithInMemorySearch", () => {
  test("search returns written documents ranked by similarity", async () => {
    const ops = new MapDocumentOps(mockEmbeddingModel());
    await ops.write({ path: "hello.txt", content: "hello world" });
    await ops.write({ path: "goodbye.txt", content: "goodbye world" });

    const results = await ops.search({ query: "hello" });
    expect(results.length).toBe(2);
    expect(results[0]!.score).toBeGreaterThanOrEqual(results[1]!.score);
  });

  test("search excludes deleted documents", async () => {
    const ops = new MapDocumentOps(mockEmbeddingModel());
    await ops.write({ path: "a.txt", content: "alpha" });
    await ops.write({ path: "b.txt", content: "beta" });
    await ops.delete({ path: "a.txt" });

    const results = await ops.search({ query: "alpha" });
    expect(results.every((r) => r.path !== "a.txt")).toBe(true);
  });

  test("search filters by path prefix", async () => {
    const ops = new MapDocumentOps(mockEmbeddingModel());
    await ops.write({ path: "docs/a.txt", content: "alpha" });
    await ops.write({ path: "src/b.txt", content: "beta" });

    const results = await ops.search({ query: "test", path: "docs" });
    expect(results.length).toBe(1);
    expect(results[0]!.path).toBe("docs/a.txt");
  });

  test("search respects limit", async () => {
    const ops = new MapDocumentOps(mockEmbeddingModel());
    await ops.write({ path: "a.txt", content: "alpha" });
    await ops.write({ path: "b.txt", content: "beta" });
    await ops.write({ path: "c.txt", content: "gamma" });

    const results = await ops.search({ query: "test", limit: 2 });
    expect(results.length).toBe(2);
  });

  test("search re-indexes after edit", async () => {
    const ops = new MapDocumentOps(mockEmbeddingModel());
    await ops.write({ path: "a.txt", content: "cats and dogs" });
    await ops.edit({ path: "a.txt", old: "dogs", new: "birds" });

    const results = await ops.search({ query: "birds" });
    expect(results[0]!.content).toBe("cats and birds");
  });
});

// –
// getSystemPrompt
// –

describe("getSystemPrompt", () => {
  test("returns instructions alone when no documents", async () => {
    const ops = new StubDocumentOps(new Map());
    const result = await ops.getSystemPrompt({ instructions: "Be helpful.", messages: [] });
    expect(result).toBe("Be helpful.");
  });

  test("returns empty string when nothing provided", async () => {
    const ops = new StubDocumentOps(new Map());
    const result = await ops.getSystemPrompt({ messages: [] });
    expect(result).toBe("");
  });

  test("injects always-inject documents", async () => {
    const ops = new StubDocumentOps(new Map([["guide.md", "The guide."]]));
    const result = await ops.getSystemPrompt({
      instructions: "Hello",
      alwaysInject: ["guide.md"],
      messages: [],
    });
    expect(result).toContain("guide.md");
    expect(result).toContain("The guide.");
    expect(result).toContain("<documents>");
  });

  test("skips unmatched always-inject globs", async () => {
    const ops = new StubDocumentOps(new Map());
    const result = await ops.getSystemPrompt({
      alwaysInject: ["missing.md"],
      messages: [],
    });
    expect(result).not.toContain("<documents>");
  });

  test("alwaysInject resolves globs", async () => {
    const ops = new StubDocumentOps(new Map([
      ["docs/a.md", "Alpha"],
      ["docs/b.md", "Beta"],
      ["src/c.ts", "Gamma"],
    ]));
    const result = await ops.getSystemPrompt({
      alwaysInject: ["*.md"],
      messages: [],
    });
    expect(result).toContain("docs/a.md");
    expect(result).toContain("docs/b.md");
    expect(result).not.toContain("src/c.ts");
  });

  test("neverInject excludes from always-inject", async () => {
    const ops = new StubDocumentOps(new Map([
      ["guide.md", "The guide"],
      ["secret.md", "Secret stuff"],
    ]));
    const result = await ops.getSystemPrompt({
      alwaysInject: ["*.md"],
      neverInject: ["secret.md"],
      messages: [],
    });
    expect(result).toContain("guide.md");
    expect(result).not.toContain("secret.md");
  });

  test("neverInject excludes from search results", async () => {
    const ops = new StubDocumentOps(
      new Map([
        ["public.md", "Public content"],
        ["secret.md", "Secret content"],
      ]),
      [
        { content: "Public content", path: "public.md", score: 0.9 },
        { content: "Secret content", path: "secret.md", score: 0.8 },
      ],
    );
    const result = await ops.getSystemPrompt({
      neverInject: ["secret.md"],
      messages: [{ role: "user", content: "find something" }],
    });
    expect(result).toContain("public.md");
    expect(result).not.toContain("secret.md");
  });

  test("includes search results from user messages", async () => {
    const ops = new StubDocumentOps(
      new Map([["result.md", "Found content"]]),
      [{ content: "Found content", path: "result.md", score: 0.9 }],
    );
    const result = await ops.getSystemPrompt({
      messages: [{ role: "user", content: "find something" }],
    });
    expect(result).toContain("result.md");
    expect(result).toContain("Found content");
  });

  test("deduplicates always-inject and search results", async () => {
    const ops = new StubDocumentOps(
      new Map([["guide.md", "The guide"]]),
      [{ content: "The guide", path: "guide.md", score: 0.8 }],
    );
    const result = await ops.getSystemPrompt({
      alwaysInject: ["guide.md"],
      messages: [{ role: "user", content: "question" }],
    });
    // path appears once in document tag, not duplicated
    expect(result.match(/path="guide\.md"/g)?.length).toBe(1);
  });

  test("truncates documents exceeding per-document budget", async () => {
    const ops = new StubDocumentOps(new Map([["big.md", "x".repeat(500)]]));
    const result = await ops.getSystemPrompt({
      alwaysInject: ["big.md"],
      messages: [],
      budgetPerDocument: 100,
    });
    expect(result).toContain("truncated");
    expect(result).toContain("chars truncated");
  });

  test("respects total budget", async () => {
    const ops = new StubDocumentOps(
      new Map([
        ["a.md", "a".repeat(200)],
        ["b.md", "b".repeat(200)],
      ]),
    );
    const result = await ops.getSystemPrompt({
      alwaysInject: ["a.md", "b.md"],
      messages: [],
      budget: 300,
    });
    // Both docs can't fit in full; at least one must be truncated or omitted
    const aCount = (result.match(/a\.md/g) || []).length;
    const bCount = (result.match(/b\.md/g) || []).length;
    expect(aCount).toBeGreaterThanOrEqual(1);
    // b.md either truncated or absent
    if (bCount > 0) {
      expect(result).toContain("truncated");
    }
  });

  test("extracts query from both user and assistant messages", async () => {
    const ops = new StubDocumentOps(
      new Map([["a.md", "content"]]),
      [{ content: "content", path: "a.md", score: 0.5 }],
    );
    const result = await ops.getSystemPrompt({
      messages: [
        { role: "assistant", content: "I can help" },
        { role: "user", content: "tell me more" },
      ] satisfies ModelMessage[],
    });
    expect(result).toContain("a.md");
  });

  test("handles array content parts in messages", async () => {
    const ops = new StubDocumentOps(
      new Map([["a.md", "content"]]),
      [{ content: "content", path: "a.md", score: 0.5 }],
    );
    const result = await ops.getSystemPrompt({
      messages: [
        { role: "user", content: [{ type: "text" as const, text: "query here" }] },
      ] satisfies ModelMessage[],
    });
    expect(result).toContain("a.md");
  });
});

// –
// getTools
// –

describe("getTools", () => {
  test("returns all document tool names", () => {
    const ops = new DocumentOps();
    const tools = ops.getTools();
    expect(Object.keys(tools).sort()).toEqual(
      ["delete", "edit", "glob", "grep", "read", "search", "write"],
    );
  });

  test("tools delegate to ops methods", async () => {
    const docs = new Map([["a.txt", "hello"]]);
    const ops = new StubDocumentOps(docs);
    const tools = ops.getTools();
    const result = await tools.read.execute!({ path: "a.txt" }, { messages: [], toolCallId: "t" });
    expect(result).toEqual({ content: "hello", lines: 1 });
  });
});
