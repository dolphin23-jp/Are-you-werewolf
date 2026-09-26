import { afterEach, describe, expect, it, vi } from "vitest";
import { sendChat } from "../api/client";

describe("API errors", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("turns a FastAPI validation error list into readable text", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        json: () =>
          Promise.resolve({ detail: [{ msg: "List should have at most 10 items" }] }),
      }),
    );

    await expect(sendChat("s1", "hi")).rejects.toThrow(
      "入力が正しくありません: List should have at most 10 items",
    );
  });
});
