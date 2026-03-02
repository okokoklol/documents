import { defineConfig } from "bunup";

export default defineConfig({
  entry: ["index.ts", "bun-sqlite.ts", "bun-directory.ts"],
  dts: { inferTypes: true },
  external: ["bun:sqlite"],
});
