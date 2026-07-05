import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import App from "./App";

describe("App", () => {
  test("renders the student mining workspace", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "Student Performance Mining UTH" })).toBeInTheDocument();
    expect(screen.getByText("Model accuracy")).toBeInTheDocument();
    expect(screen.getByText("25,000")).toBeInTheDocument();
    expect(screen.getByLabelText("Age")).toBeInTheDocument();
    expect(screen.getByLabelText("Study hours")).toBeInTheDocument();
    expect(screen.getByLabelText("Math score")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /predict grade/i })).toBeInTheDocument();
  });
});
