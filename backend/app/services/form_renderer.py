from __future__ import annotations

from html import escape


def _format_probability(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def render_form(bundle: dict[str, object], result: dict[str, object] | None = None) -> str:
    fields_html: list[str] = []
    for field in bundle["input_schema"]:
        name = str(field["name"])
        label = name.replace("_", " ").title()
        if field["kind"] == "category":
            options = "".join(
                f'<option value="{escape(str(option))}">{escape(str(option))}</option>'
                for option in field["options"]
            )
            control = f'<select name="{escape(name)}" required>{options}</select>'
        else:
            control = (
                f'<input name="{escape(name)}" type="number" step="any" '
                f'placeholder="{escape(label)}" required>'
            )
        fields_html.append(f"<label><span>{escape(label)}</span>{control}</label>")

    result_html = ""
    if result:
        probabilities = result["probabilities"]
        probability_rows = "".join(
            "<li>"
            f"<strong>{escape(str(label))}</strong>: {_format_probability(float(probability))}"
            "</li>"
            for label, probability in probabilities.items()
        )
        result_html = (
            '<section class="result">'
            f'<h2>Du doan: {escape(str(result["predicted_grade"]))}</h2>'
            f'<p>Ma diem: {escape(str(result["predicted_grade_code"]))}</p>'
            f"<ul>{probability_rows}</ul>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Student Grade Prediction</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      background: #f5f7fb;
      color: #172033;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 32px 20px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    p {{
      margin: 0 0 24px;
      color: #526071;
    }}
    form {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-weight: 600;
      font-size: 14px;
    }}
    input, select {{
      border: 1px solid #c9d2df;
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 15px;
      background: white;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      padding: 12px 16px;
      font-size: 15px;
      font-weight: 700;
      color: white;
      background: #2463eb;
      cursor: pointer;
    }}
    .result {{
      margin-top: 28px;
      padding: 18px;
      border: 1px solid #c9d2df;
      border-radius: 8px;
      background: white;
    }}
    .result h2 {{
      margin: 0 0 10px;
      font-size: 22px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Du doan ket qua sinh vien</h1>
    <p>Nhap thong tin sinh vien, backend se du doan final_grade bang model da train.</p>
    <form method="post" action="/predict-form">
      {''.join(fields_html)}
      <button type="submit">Du doan</button>
    </form>
    {result_html}
  </main>
</body>
</html>"""
