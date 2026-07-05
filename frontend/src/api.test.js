import { describe, expect, test, vi } from "vitest";

import { predictStudent } from "./api";

describe("predictStudent", () => {
  test("posts student data to the backend predict endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ predicted_grade: "Khá", predicted_grade_code: "c" }),
    });

    const result = await predictStudent({ age: 17 }, fetchMock);

    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ age: 17 }),
    });
    expect(result).toEqual({ predicted_grade: "Khá", predicted_grade_code: "c" });
  });
});
