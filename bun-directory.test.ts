import { afterAll, beforeEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MockEmbeddingModelV3 } from "ai/test";
import BunDirectoryDocumentOps from "./bun-directory";

// –
// Helpers
// –

function hashToVector(text: string, dim = 16): number[] {
  const vec = new Array(dim).fill(0);
  for (let i = 0; i < text.length; i++) vec[i % dim] += text.charCodeAt(i);
  const norm = Math.sqrt(vec.reduce((s: number, v: number) => s + v * v, 0)) || 1;
  return vec.map((v: number) => v / norm);
}

let root: string;
let ops: BunDirectoryDocumentOps;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "harness-test-"));
  ops = new BunDirectoryDocumentOps({
    root,
    embeddingModel: new MockEmbeddingModelV3({
      doEmbed: async ({ values }) => ({
        embeddings: values.map((v) => hashToVector(v)),
        warnings: [],
      }),
    }),
  });
});

afterAll(async () => {
  if (root) await rm(root, { recursive: true, force: true });
});

// –
// Write / Read
// –

describe("write + read", () => {
  test("roundtrip", async () => {
    await ops.write({ path: "hello.txt", content: "hello world" });
    const result = await ops.read({ path: "hello.txt" });
    expect(result.content).toBe("hello world");
    expect(result.lines).toBe(1);
  });

  test("creates nested directories", async () => {
    await ops.write({ path: "deep/nested/file.txt", content: "ok" });
    expect((await ops.read({ path: "deep/nested/file.txt" })).content).toBe("ok");
  });

  test("overwrite replaces content", async () => {
    await ops.write({ path: "a.txt", content: "original" });
    await ops.write({ path: "a.txt", content: "updated" });
    expect((await ops.read({ path: "a.txt" })).content).toBe("updated");
  });

  test("read with offset and limit", async () => {
    await ops.write({ path: "a.txt", content: "L1\nL2\nL3\nL4" });
    const result = await ops.read({ path: "a.txt", offset: 2, limit: 2 });
    expect(result.content).toBe("L2\nL3");
    expect(result.lines).toBe(4);
  });

  test("read throws on missing path", async () => {
    expect(ops.read({ path: "missing.txt" })).rejects.toThrow();
  });
});

// –
// Path jailing
// –

describe("path jailing", () => {
  test("rejects path that escapes root", async () => {
    expect(ops.read({ path: "../etc/passwd" })).rejects.toThrow(/escapes root/i);
  });

  test("rejects absolute path outside root", async () => {
    expect(ops.read({ path: "/etc/passwd" })).rejects.toThrow(/escapes root/i);
  });
});

// –
// Edit
// –

describe("edit", () => {
  test("unique replacement", async () => {
    await ops.write({ path: "a.txt", content: "foo bar baz" });
    await ops.edit({ path: "a.txt", old: "bar", new: "qux" });
    expect((await ops.read({ path: "a.txt" })).content).toBe("foo qux baz");
  });

  test("replace all occurrences", async () => {
    await ops.write({ path: "a.txt", content: "aaa" });
    await ops.edit({ path: "a.txt", old: "a", new: "b", all: true });
    expect((await ops.read({ path: "a.txt" })).content).toBe("bbb");
  });

  test("throws on identical replacement", async () => {
    await ops.write({ path: "a.txt", content: "foo" });
    expect(ops.edit({ path: "a.txt", old: "foo", new: "foo" })).rejects.toThrow(/identical/i);
  });

  test("throws when string not found", async () => {
    await ops.write({ path: "a.txt", content: "foo" });
    expect(ops.edit({ path: "a.txt", old: "bar", new: "baz" })).rejects.toThrow(/not found/i);
  });

  test("throws on ambiguous match", async () => {
    await ops.write({ path: "a.txt", content: "foo foo" });
    expect(ops.edit({ path: "a.txt", old: "foo", new: "bar" })).rejects.toThrow(/ambiguous/i);
  });
});

// –
// Delete
// –

describe("delete", () => {
  test("removes file", async () => {
    await ops.write({ path: "a.txt", content: "hello" });
    await ops.delete({ path: "a.txt" });
    expect(ops.read({ path: "a.txt" })).rejects.toThrow();
  });

  test("throws on missing path", async () => {
    expect(ops.delete({ path: "missing.txt" })).rejects.toThrow();
  });
});

// –
// Glob
// –

describe("glob", () => {
  test("matches by extension", async () => {
    await ops.write({ path: "src/a.ts", content: "" });
    await ops.write({ path: "src/b.ts", content: "" });
    await ops.write({ path: "readme.md", content: "" });

    const results = await ops.glob({ pattern: "*.ts" });
    expect(results.sort()).toEqual(["src/a.ts", "src/b.ts"]);
  });

  test("respects limit", async () => {
    await ops.write({ path: "a.txt", content: "" });
    await ops.write({ path: "b.txt", content: "" });
    await ops.write({ path: "c.txt", content: "" });
    expect((await ops.glob({ pattern: "*.txt", limit: 2 })).length).toBe(2);
  });

  test("scopes by path prefix", async () => {
    await ops.write({ path: "src/a.ts", content: "" });
    await ops.write({ path: "lib/b.ts", content: "" });

    const results = await ops.glob({ pattern: "*.ts", path: "src" });
    expect(results).toEqual(["a.ts"]);
  });
});

// –
// Grep
// –

describe("grep", () => {
  test("finds matching lines", async () => {
    await ops.write({ path: "a.txt", content: "hello world\ngoodbye world" });

    const matches = await ops.grep({ pattern: "hello" });
    expect(matches.length).toBe(1);
    expect(matches[0]!.content).toBe("hello world");
    expect(matches[0]!.line).toBe(1);
    expect(matches[0]!.path).toBe("a.txt");
  });

  test("case-insensitive", async () => {
    await ops.write({ path: "a.txt", content: "Hello World" });
    expect((await ops.grep({ pattern: "hello", ignoreCase: true })).length).toBe(1);
    expect((await ops.grep({ pattern: "hello" })).length).toBe(0);
  });

  test("includes context lines", async () => {
    await ops.write({ path: "a.txt", content: "alpha\nbeta\ngamma\ndelta\nepsilon" });

    const matches = await ops.grep({ pattern: "gamma", before: 1, after: 1 });
    expect(matches[0]!.content).toBe("beta\ngamma\ndelta");
  });

  test("filters by file glob", async () => {
    await ops.write({ path: "a.ts", content: "match" });
    await ops.write({ path: "b.md", content: "match" });

    const matches = await ops.grep({ pattern: "match", fileGlob: "*.ts" });
    expect(matches.length).toBe(1);
    expect(matches[0]!.path).toBe("a.ts");
  });

  test("respects limit", async () => {
    await ops.write({ path: "a.txt", content: "match\nmatch\nmatch" });
    expect((await ops.grep({ pattern: "match", limit: 2 })).length).toBe(2);
  });

  test("respects offset", async () => {
    await ops.write({ path: "a.txt", content: "match1\nmatch2\nmatch3" });

    const matches = await ops.grep({ pattern: "match", offset: 1 });
    expect(matches.length).toBe(2);
    expect(matches[0]!.line).toBe(2);
  });

  test("scopes by path prefix", async () => {
    await ops.write({ path: "src/a.txt", content: "found" });
    await ops.write({ path: "lib/b.txt", content: "found" });

    const matches = await ops.grep({ pattern: "found", path: "src" });
    expect(matches.length).toBe(1);
  });

  test("multiline matching", async () => {
    await ops.write({ path: "a.txt", content: "start\nmiddle\nend" });

    const matches = await ops.grep({ pattern: "start.*middle", multiline: true });
    expect(matches.length).toBe(1);
  });
});
