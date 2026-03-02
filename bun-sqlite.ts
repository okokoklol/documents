/**
 * DocumentOpsWithInMemorySearch backed by Bun SQLite.
 * All documents live in a single `documents` table keyed by path.
 */

import type { EmbeddingModel } from "ai";
import { type Database } from "bun:sqlite";
import { DocumentOpsWithInMemorySearch } from "./index";

export default class BunSQLiteDocumentOps extends DocumentOpsWithInMemorySearch {
  private readonly ops: ReturnType<typeof sqliteOperations>;

  constructor(params: {
    /** Bun SQLite database instance. */
    database: Database;
    /** Embedding model for in-memory search. */
    embeddingModel: EmbeddingModel;
    /** Max characters to embed per document. */
    embeddingLimit?: number;
  }) {
    super(params.embeddingModel, params.embeddingLimit);
    this.ops = sqliteOperations(params.database);
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
// SQLite Operations
// –

interface Row {
  /** File content. */
  content: string;
  /** File path (primary key). */
  path: string;
}

function sqliteOperations(db: Database) {
  db.run("CREATE TABLE IF NOT EXISTS documents (path TEXT PRIMARY KEY, content TEXT NOT NULL)");

  const stmts = {
    all: db.query<Row, []>("SELECT path, content FROM documents ORDER BY path"),
    delete: db.query<null, { $path: string }>("DELETE FROM documents WHERE path = $path"),
    paths: db.query<Pick<Row, "path">, []>("SELECT path FROM documents ORDER BY path"),
    prefixed: db.query<Row, { $prefix: string }>(
      "SELECT path, content FROM documents WHERE path LIKE $prefix ORDER BY path",
    ),
    prefixedPaths: db.query<Pick<Row, "path">, { $prefix: string }>(
      "SELECT path FROM documents WHERE path LIKE $prefix ORDER BY path",
    ),
    read: db.query<Pick<Row, "content">, { $path: string }>(
      "SELECT content FROM documents WHERE path = $path",
    ),
    write: db.query<null, { $path: string; $content: string }>(
      "INSERT OR REPLACE INTO documents (path, content) VALUES ($path, $content)",
    ),
  };

  const load = (path: string): string => {
    const row = stmts.read.get({ $path: path });
    if (!row) throw new Error(`File not found: ${path}`);
    return row.content;
  };

  const likePrefix = (path: string) =>
    (path.endsWith("/") ? path : path + "/") + "%";

  const stripPrefix = (stored: string, dir: string) => {
    const p = dir.endsWith("/") ? dir : dir + "/";
    return stored.startsWith(p) ? stored.slice(p.length) : stored;
  };

  return {
    async delete({ path }: { path: string }) {
      if (!stmts.read.get({ $path: path })) throw new Error(`File not found: ${path}`);
      stmts.delete.run({ $path: path });
    },

    async edit({ path, old, new: replacement, all }: {
      path: string; old: string; new: string; all?: boolean;
    }) {
      const content = load(path);
      if (old === replacement) throw new Error("Replacement is identical to original");
      if (!content.includes(old)) throw new Error("String not found");

      if (all) {
        stmts.write.run({ $path: path, $content: content.replaceAll(old, replacement) });
        return;
      }

      if (content.indexOf(old) !== content.lastIndexOf(old))
        throw new Error("String is ambiguous (found multiple occurrences)");

      const i = content.indexOf(old);
      stmts.write.run({
        $path: path,
        $content: content.slice(0, i) + replacement + content.slice(i + old.length),
      });
    },

    async glob({ pattern, path, limit }: { pattern: string; path?: string; limit?: number }) {
      const effective = pattern.startsWith("**/") ? pattern : `**/${pattern}`;
      const g = new Bun.Glob(effective);

      const rows = path
        ? stmts.prefixedPaths.all({ $prefix: likePrefix(path) })
        : stmts.paths.all();

      const results: string[] = [];
      for (const row of rows) {
        const relative = path ? stripPrefix(row.path, path) : row.path;
        if (g.match(relative)) {
          results.push(relative);
          if (limit && results.length >= limit) break;
        }
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
      const rows = params.path
        ? stmts.prefixed.all({ $prefix: likePrefix(params.path) })
        : stmts.all.all();

      const fileGlob = params.fileGlob
        ? new Bun.Glob(params.fileGlob.startsWith("**/") ? params.fileGlob : `**/${params.fileGlob}`)
        : null;
      const flags = `g${params.ignoreCase ? "i" : ""}${params.multiline ? "ms" : ""}`;
      const regex = new RegExp(params.pattern, flags);
      const ctxBefore = params.before ?? 0;
      const ctxAfter = params.after ?? 0;
      const skip = params.offset ?? 0;

      const matches: { content: string; line: number; path: string }[] = [];
      let skipped = 0;

      for (const row of rows) {
        if (params.limit && matches.length >= params.limit) break;
        if (fileGlob && !fileGlob.match(row.path)) continue;

        const lines = row.content.split("\n");

        if (params.multiline) {
          regex.lastIndex = 0;
          let m: RegExpExecArray | null;
          while ((m = regex.exec(row.content)) !== null) {
            if (skipped++ < skip) continue;
            const ln = row.content.slice(0, m.index).split("\n").length;
            const lo = Math.max(0, ln - 1 - ctxBefore);
            const hi = Math.min(lines.length, ln + ctxAfter);
            matches.push({ content: lines.slice(lo, hi).join("\n"), line: ln, path: row.path });
            if (params.limit && matches.length >= params.limit) break;
          }
        } else {
          for (let i = 0; i < lines.length; i++) {
            regex.lastIndex = 0;
            if (!regex.test(lines[i]!)) continue;
            if (skipped++ < skip) continue;
            const lo = Math.max(0, i - ctxBefore);
            const hi = Math.min(lines.length, i + 1 + ctxAfter);
            matches.push({ content: lines.slice(lo, hi).join("\n"), line: i + 1, path: row.path });
            if (params.limit && matches.length >= params.limit) break;
          }
        }
      }

      return matches;
    },

    async read({ path, offset, limit }: { path: string; offset?: number; limit?: number }) {
      const content = load(path);
      const allLines = content.split("\n");
      const total = content.length === 0 ? 0 : allLines.length;

      if (offset !== undefined || limit !== undefined) {
        const start = (offset ?? 1) - 1;
        const end = limit !== undefined ? start + limit : total;
        return { content: allLines.slice(start, end).join("\n"), lines: total };
      }

      return { content, lines: total };
    },

    async write({ path, content }: { path: string; content: string }) {
      stmts.write.run({ $path: path, $content: content });
    },
  };
}
