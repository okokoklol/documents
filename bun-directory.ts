/**
 * DocumentOpsWithInMemorySearch backed by the local filesystem via Bun.
 * All paths resolve relative to `root` and cannot escape it.
 */

import type { EmbeddingModel } from "ai";
import { mkdir, unlink } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { DocumentOpsWithInMemorySearch } from "./index";

export default class BunDirectoryDocumentOps extends DocumentOpsWithInMemorySearch {
  private readonly ops: ReturnType<typeof directoryOperations>;

  constructor(params: {
    /** Absolute path to the root documents directory. */
    root: string;
    /** Embedding model for in-memory search. */
    embeddingModel: EmbeddingModel;
    /** Max characters to embed per document. */
    embeddingLimit?: number;
  }) {
    super(params.embeddingModel, params.embeddingLimit);
    this.ops = directoryOperations(params.root);
  }

  override async delete(params: { path: string }): Promise<void> {
    await this.ops.delete(params);
    super.delete(params);
  }

  override async edit(params: {
    path: string;
    old: string;
    new: string;
    all?: boolean;
  }): Promise<void> {
    await this.ops.edit(params);
    super.edit(params);
  }

  override async glob(params: {
    pattern: string;
    path?: string;
    limit?: number;
  }): Promise<string[]> {
    return this.ops.glob(params);
  }

  override async grep(params: Parameters<typeof this.ops.grep>[0]): Promise<
    { content: string; line: number; path: string }[]
  > {
    return this.ops.grep(params);
  }

  override async read(params: {
    path: string;
    offset?: number;
    limit?: number;
  }): Promise<{ content: string; lines: number }> {
    return this.ops.read(params);
  }

  override async write(params: { path: string; content: string }): Promise<void> {
    await this.ops.write(params);
    super.write(params);
  }
}

// –
// Directory Operations
// –

function directoryOperations(root: string) {
  const normalized = resolve(root);

  /** Resolve and jail a relative path within root. */
  const abs = (p: string) => {
    const resolved = resolve(normalized, p);
    if (resolved !== normalized && !resolved.startsWith(normalized + "/"))
      throw new Error(`Path escapes root: ${p}`);
    return resolved;
  };

  const rel = (p: string) => relative(normalized, p);

  return {
    async delete({ path }: { path: string }) {
      await unlink(abs(path));
    },

    async edit({ path, old, new: replacement, all }: {
      path: string; old: string; new: string; all?: boolean;
    }) {
      const full = abs(path);
      const content = await Bun.file(full).text();

      if (old === replacement) throw new Error("Replacement is identical to original");
      if (!content.includes(old)) throw new Error("String not found");

      if (all) {
        await Bun.write(full, content.replaceAll(old, replacement));
        return;
      }

      if (content.indexOf(old) !== content.lastIndexOf(old))
        throw new Error("String is ambiguous (found multiple occurrences)");

      const i = content.indexOf(old);
      await Bun.write(full, content.slice(0, i) + replacement + content.slice(i + old.length));
    },

    async glob({ pattern, path, limit }: { pattern: string; path?: string; limit?: number }) {
      const cwd = path ? abs(path) : normalized;
      const effective = pattern.startsWith("**/") ? pattern : `**/${pattern}`;
      const g = new Bun.Glob(effective);
      const results: string[] = [];
      for await (const match of g.scan({ cwd })) {
        results.push(match);
        if (limit && results.length >= limit) break;
      }
      return results;
    },

    async grep(params: {
      after?: number;
      before?: number;
      fileGlob?: string;
      ignoreCase?: boolean;
      limit?: number;
      multiline?: boolean;
      offset?: number;
      path?: string;
      pattern: string;
    }) {
      const cwd = params.path ? abs(params.path) : normalized;
      const filePattern = new Bun.Glob(params.fileGlob ?? "**/*");
      const flags = `g${params.ignoreCase ? "i" : ""}${params.multiline ? "ms" : ""}`;
      const regex = new RegExp(params.pattern, flags);
      const ctxBefore = params.before ?? 0;
      const ctxAfter = params.after ?? 0;
      const skip = params.offset ?? 0;

      const matches: { content: string; line: number; path: string }[] = [];
      let skipped = 0;

      for await (const file of filePattern.scan({ cwd, onlyFiles: true })) {
        if (params.limit && matches.length >= params.limit) break;

        const fullPath = resolve(cwd, file);
        let text: string;
        try {
          text = await Bun.file(fullPath).text();
        } catch {
          continue;
        }

        const lines = text.split("\n");

        if (params.multiline) {
          regex.lastIndex = 0;
          let m: RegExpExecArray | null;
          while ((m = regex.exec(text)) !== null) {
            if (skipped++ < skip) continue;
            const ln = text.slice(0, m.index).split("\n").length;
            const lo = Math.max(0, ln - 1 - ctxBefore);
            const hi = Math.min(lines.length, ln + ctxAfter);
            matches.push({ content: lines.slice(lo, hi).join("\n"), line: ln, path: rel(fullPath) });
            if (params.limit && matches.length >= params.limit) break;
          }
        } else {
          for (let i = 0; i < lines.length; i++) {
            regex.lastIndex = 0;
            if (!regex.test(lines[i]!)) continue;
            if (skipped++ < skip) continue;
            const lo = Math.max(0, i - ctxBefore);
            const hi = Math.min(lines.length, i + 1 + ctxAfter);
            matches.push({ content: lines.slice(lo, hi).join("\n"), line: i + 1, path: rel(fullPath) });
            if (params.limit && matches.length >= params.limit) break;
          }
        }
      }

      return matches;
    },

    async read({ path, offset, limit }: { path: string; offset?: number; limit?: number }) {
      const text = await Bun.file(abs(path)).text();
      const allLines = text.split("\n");
      const total = text.length === 0 ? 0 : allLines.length;

      if (offset !== undefined || limit !== undefined) {
        const start = (offset ?? 1) - 1;
        const end = limit !== undefined ? start + limit : total;
        return { content: allLines.slice(start, end).join("\n"), lines: total };
      }

      return { content: text, lines: total };
    },

    async write({ path, content }: { path: string; content: string }) {
      const target = abs(path);
      await mkdir(dirname(target), { recursive: true });
      await Bun.write(target, content);
    },
  };
}
