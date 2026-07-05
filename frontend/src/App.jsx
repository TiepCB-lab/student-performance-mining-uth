import { Activity, BarChart3, GraduationCap, Loader2, Send } from "lucide-react";
import { useMemo, useState } from "react";

import { predictStudent } from "./api";

const gradeDistribution = [
  { grade: "A", label: "Xuất sắc", count: 1205, color: "#237a57" },
  { grade: "B", label: "Giỏi", count: 2696, color: "#2f6fb0" },
  { grade: "C", label: "Khá", count: 6161, color: "#8a5f11" },
  { grade: "D", label: "Trung bình", count: 6311, color: "#9b4f18" },
  { grade: "E", label: "Yếu", count: 5672, color: "#9b334f" },
  { grade: "F", label: "Kém", count: 2955, color: "#6f4aa8" },
];

const initialForm = {
  age: 17,
  gender: "female",
  school_type: "public",
  parent_education: "graduate",
  study_hours: 4.5,
  attendance_percentage: 88,
  internet_access: "yes",
  travel_time: "15-30 min",
  extra_activities: "yes",
  study_method: "notes",
  math_score: 72,
  science_score: 78,
  english_score: 75,
};

const selectFields = [
  { name: "gender", label: "Gender", options: ["female", "male", "other"] },
  { name: "school_type", label: "School type", options: ["public", "private"] },
  {
    name: "parent_education",
    label: "Parent education",
    options: ["high school", "diploma", "graduate", "post graduate"],
  },
  { name: "internet_access", label: "Internet access", options: ["yes", "no"] },
  { name: "travel_time", label: "Travel time", options: ["<15 min", "15-30 min", "30-60 min", ">60 min"] },
  { name: "extra_activities", label: "Extra activities", options: ["yes", "no"] },
  { name: "study_method", label: "Study method", options: ["notes", "textbook", "video", "group study"] },
];

const numberFields = [
  { name: "age", label: "Age", min: 13, max: 22, step: 1 },
  { name: "study_hours", label: "Study hours", min: 0, max: 12, step: 0.1 },
  { name: "attendance_percentage", label: "Attendance percentage", min: 0, max: 100, step: 0.1 },
  { name: "math_score", label: "Math score", min: 0, max: 100, step: 0.1 },
  { name: "science_score", label: "Science score", min: 0, max: 100, step: 0.1 },
  { name: "english_score", label: "English score", min: 0, max: 100, step: 0.1 },
];

function Metric({ icon: Icon, label, value, detail }) {
  return (
    <article className="metric">
      <Icon aria-hidden="true" size={20} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </article>
  );
}

function Field({ field, value, onChange }) {
  return (
    <label className="field">
      <span>{field.label}</span>
      <input
        name={field.name}
        type="number"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        onChange={onChange}
      />
    </label>
  );
}

function SelectField({ field, value, onChange }) {
  return (
    <label className="field">
      <span>{field.label}</span>
      <select name={field.name} value={value} onChange={onChange}>
        {field.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const totalStudents = useMemo(
    () => gradeDistribution.reduce((sum, item) => sum + item.count, 0),
    []
  );

  function updateField(event) {
    const { name, value, type } = event.target;
    setForm((current) => ({
      ...current,
      [name]: type === "number" ? Number(value) : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const prediction = await predictStudent(form);
      setResult(prediction);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="workspace">
      <section className="topbar">
        <div>
          <p className="eyebrow">UTH data mining workspace</p>
          <h1>Student Performance Mining UTH</h1>
        </div>
        <span className="status">FastAPI connected at port 8000</span>
      </section>

      <section className="metrics" aria-label="Model summary">
        <Metric icon={Activity} label="Model accuracy" value="84.53%" detail="LightGBM test split" />
        <Metric icon={GraduationCap} label="Training rows" value="18,750" detail="25% test holdout" />
        <Metric icon={BarChart3} label="Dataset size" value="25,000" detail="Student records" />
      </section>

      <section className="content-grid">
        <form className="predictor" onSubmit={handleSubmit}>
          <div className="section-title">
            <h2>Prediction Input</h2>
            <span>13 features</span>
          </div>

          <div className="form-grid">
            {numberFields.map((field) => (
              <Field key={field.name} field={field} value={form[field.name]} onChange={updateField} />
            ))}
            {selectFields.map((field) => (
              <SelectField key={field.name} field={field} value={form[field.name]} onChange={updateField} />
            ))}
          </div>

          <button className="primary-action" type="submit" disabled={loading}>
            {loading ? <Loader2 aria-hidden="true" className="spin" size={18} /> : <Send aria-hidden="true" size={18} />}
            Predict grade
          </button>

          {error && <p className="error">{error}</p>}
        </form>

        <aside className="insights">
          <section className="result-panel">
            <div className="section-title">
              <h2>Prediction Result</h2>
              <span>{result ? result.predicted_grade_code.toUpperCase() : "Ready"}</span>
            </div>
            {result ? (
              <div className="grade-result">
                <strong>{result.predicted_grade}</strong>
                <p>Predicted grade code: {result.predicted_grade_code.toUpperCase()}</p>
                <div className="probabilities">
                  {Object.entries(result.probabilities || {}).map(([label, probability]) => (
                    <div key={label}>
                      <span>{label}</span>
                      <meter min="0" max="1" value={probability} />
                      <small>{(probability * 100).toFixed(1)}%</small>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="empty-state">Submit a student profile to see the predicted final grade.</p>
            )}
          </section>

          <section className="distribution">
            <div className="section-title">
              <h2>Grade Distribution</h2>
              <span>{totalStudents.toLocaleString()} rows</span>
            </div>
            {gradeDistribution.map((item) => (
              <div className="bar-row" key={item.grade}>
                <span>{item.grade}</span>
                <div>
                  <i style={{ width: `${(item.count / totalStudents) * 100}%`, backgroundColor: item.color }} />
                </div>
                <strong>{item.count.toLocaleString()}</strong>
              </div>
            ))}
          </section>
        </aside>
      </section>
    </main>
  );
}
