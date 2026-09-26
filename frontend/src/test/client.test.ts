import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getView, sendChat } from "../api/client";

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

  it("keeps the HTTP status so a missing game can be told from a network blip", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: "session not found" }),
      }),
    );

    const error = await getView("s1", "p0").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(404);
    expect((error as ApiError).message).toBe("session not found");
  });
});
