import { useState, useEffect } from "react";
import {
  User, BookOpen, Heart, ChevronRight, ChevronLeft,
  Brain, BarChart2, Loader2, AlertCircle, CheckCircle2,
  FlaskConical, School, RefreshCw
} from "lucide-react";

const BASE_URL = "http://localhost:8000/api";

const MOCK_DATA = {
  school: "GP", sex: "M", age: 17, address: "U", famsize: "GT3",
  Pstatus: "T", Medu: 3, Fedu: 2, Mjob: "health", Fjob: "services",
  guardian: "mother", famrel: 4, reason: "reputation", traveltime: 2,
  studytime: 3, failures: 0, schoolsup: "no", famsup: "yes",
  paid: "no", activities: "yes", higher: "yes", nursery: "yes",
  internet: "yes", romantic: "no", freetime: 3, goout: 2,
  Dalc: 1, Walc: 2, health: 4, absences: 4, G1: 14, G2: 15,
};

const INIT_FORM = {
  school: "GP", sex: "M", age: 17, address: "U", famsize: "GT3",
  Pstatus: "T", Medu: 2, Fedu: 2, Mjob: "other", Fjob: "other",
  guardian: "mother", famrel: 3, reason: "reputation", traveltime: 1,
  studytime: 2, failures: 0, schoolsup: "no", famsup: "no",
  paid: "no", activities: "no", higher: "yes", nursery: "yes",
  internet: "yes", romantic: "no", freetime: 3, goout: 3,
  Dalc: 1, Walc: 1, health: 3, absences: 0, G1: 10, G2: 11,
};

const NUM_FIELDS = ["age","Medu","Fedu","traveltime","studytime","failures","famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2"];

const gradeColor = (g) => g >= 15 ? "#22c55e" : g >= 10 ? "#f59e0b" : "#ef4444";
const gradeLabel = (g) => g >= 18 ? "Xuất sắc" : g >= 15 ? "Giỏi" : g >= 12 ? "Khá" : g >= 10 ? "Trung bình" : "Yếu";

// ── Shared field styles (inline to guarantee rendering) ──────────────────────
const S = {
  label: { color: "#94a3b8", fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: "4px" },
  input: { width: "100%", backgroundColor: "#0f1e35", border: "1px solid #1e3a5f", color: "#ffffff", borderRadius: "8px", padding: "8px 12px", fontSize: "14px", outline: "none", boxSizing: "border-box" },
  inputFocused: { width: "100%", backgroundColor: "#0f1e35", border: "1px solid #6366f1", color: "#ffffff", borderRadius: "8px", padding: "8px 12px", fontSize: "14px", outline: "none", boxSizing: "border-box" },
};

// ── CircularGauge ────────────────────────────────────────────────────────────
function CircularGauge({ score }) {
  const r = 54, circumference = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(score / 20, 1));
  const color = gradeColor(score);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
      <svg width="128" height="128" viewBox="0 0 128 128">
        <circle cx="64" cy="64" r={r} fill="none" stroke="#1e2d4a" strokeWidth="10" />
        <circle cx="64" cy="64" r={r} fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={circumference} strokeDashoffset={circumference * (1 - pct)}
          strokeLinecap="round" transform="rotate(-90 64 64)"
          style={{ transition: "stroke-dashoffset 0.8s ease" }} />
        <text x="64" y="60" textAnchor="middle" fill="white" fontSize="26" fontWeight="bold" fontFamily="monospace">
          {score !== null ? score.toFixed(1) : "—"}
        </text>
        <text x="64" y="78" textAnchor="middle" fill="#94a3b8" fontSize="11">/ 20.0</text>
      </svg>
      <span style={{ backgroundColor: color + "22", color, fontSize: "13px", fontWeight: 600, padding: "3px 12px", borderRadius: "999px" }}>
        {gradeLabel(score)}
      </span>
    </div>
  );
}

// ── Form field components ─────────────────────────────────────────────────────
function FSelect({ label, name, options, value, onChange }) {
  const [focused, setFocused] = useState(false);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <label style={S.label}>{label}</label>
      <select
        name={name} value={value} onChange={onChange}
        onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        style={focused ? S.inputFocused : S.input}
      >
        {options.map(o => (
          <option key={o.value} value={o.value} style={{ backgroundColor: "#0f1e35", color: "#fff" }}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

function FNumber({ label, name, min, max, value, onChange }) {
  const [focused, setFocused] = useState(false);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <label style={S.label}>{label}</label>
      <input
        type="number" name={name} min={min} max={max} value={value} onChange={onChange}
        onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        style={focused ? S.inputFocused : S.input}
      />
    </div>
  );
}

function FRadio({ label, name, options, value, onChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <label style={S.label}>{label}</label>
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {options.map(o => {
          const active = value === o.value;
          return (
            <label key={o.value} style={{
              display: "flex", alignItems: "center", gap: "6px",
              padding: "6px 12px", borderRadius: "8px", cursor: "pointer", fontSize: "13px",
              border: active ? "1px solid #6366f1" : "1px solid #1e3a5f",
              backgroundColor: active ? "rgba(99,102,241,0.15)" : "transparent",
              color: active ? "#a5b4fc" : "#94a3b8",
              transition: "all 0.15s",
            }}>
              <input type="radio" style={{ display: "none" }} name={name} value={o.value} checked={active} onChange={onChange} />
              {o.label}
            </label>
          );
        })}
      </div>
    </div>
  );
}

function FToggle({ label, name, value, onChange }) {
  const isYes = value === "yes";
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 0" }}>
      <span style={{ color: "#cbd5e1", fontSize: "14px" }}>{label}</span>
      <button type="button"
        onClick={() => onChange({ target: { name, value: isYes ? "no" : "yes" } })}
        style={{ position: "relative", width: "44px", height: "24px", borderRadius: "999px", border: "none", cursor: "pointer",
          backgroundColor: isYes ? "#6366f1" : "#1e3a5f", transition: "background-color 0.2s" }}>
        <span style={{ position: "absolute", top: "4px", width: "16px", height: "16px", borderRadius: "50%",
          backgroundColor: "#fff", transition: "transform 0.2s", transform: isYes ? "translateX(24px)" : "translateX(4px)" }} />
      </button>
    </div>
  );
}

function FSlider({ label, name, min, max, value, onChange }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <label style={S.label}>{label}</label>
        <span style={{ color: "#818cf8", fontSize: "12px", fontWeight: 700, fontFamily: "monospace" }}>{value}</span>
      </div>
      <input type="range" min={min} max={max} value={value} name={name} onChange={onChange}
        style={{ width: "100%", accentColor: "#6366f1" }} />
      <div style={{ display: "flex", justifyContent: "space-between", color: "#334155", fontSize: "11px" }}>
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  );
}

// ── Tab content ───────────────────────────────────────────────────────────────
const EDU_OPTS = [
  { value: 0, label: "0 – Không học" }, { value: 1, label: "1 – Tiểu học" },
  { value: 2, label: "2 – THCS" }, { value: 3, label: "3 – THPT" }, { value: 4, label: "4 – Đại học+" },
];
const JOB_OPTS = [
  { value: "teacher", label: "Giáo viên" }, { value: "health", label: "Y tế" },
  { value: "services", label: "Dịch vụ" }, { value: "at_home", label: "Ở nhà" }, { value: "other", label: "Khác" },
];

function Tab1({ form, handle }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "20px" }}>
      <FSelect label="Trường học" name="school" value={form.school} onChange={handle}
        options={[{ value: "GP", label: "GP – Gabriel Pereira" }, { value: "MS", label: "MS – Mousinho da Silveira" }]} />
      <FSelect label="Giới tính" name="sex" value={form.sex} onChange={handle}
        options={[{ value: "F", label: "Nữ (F)" }, { value: "M", label: "Nam (M)" }]} />
      <FNumber label="Tuổi (15–22)" name="age" min={15} max={22} value={form.age} onChange={handle} />
      <FRadio label="Khu vực sống" name="address" value={form.address} onChange={handle}
        options={[{ value: "U", label: "🏙 Thành thị" }, { value: "R", label: "🌾 Nông thôn" }]} />
      <FRadio label="Quy mô gia đình" name="famsize" value={form.famsize} onChange={handle}
        options={[{ value: "LE3", label: "≤ 3 người" }, { value: "GT3", label: "> 3 người" }]} />
      <FRadio label="Trạng thái bố mẹ" name="Pstatus" value={form.Pstatus} onChange={handle}
        options={[{ value: "T", label: "Sống chung" }, { value: "A", label: "Sống riêng" }]} />
      <FSelect label="Học vấn của mẹ (Medu)" name="Medu" value={form.Medu} onChange={handle} options={EDU_OPTS} />
      <FSelect label="Học vấn của bố (Fedu)" name="Fedu" value={form.Fedu} onChange={handle} options={EDU_OPTS} />
      <FSelect label="Nghề của mẹ (Mjob)" name="Mjob" value={form.Mjob} onChange={handle} options={JOB_OPTS} />
      <FSelect label="Nghề của bố (Fjob)" name="Fjob" value={form.Fjob} onChange={handle} options={JOB_OPTS} />
      <FSelect label="Người giám hộ" name="guardian" value={form.guardian} onChange={handle}
        options={[{ value: "mother", label: "Mẹ" }, { value: "father", label: "Bố" }, { value: "other", label: "Khác" }]} />
      <FSlider label="Gắn kết gia đình (famrel)" name="famrel" min={1} max={5} value={form.famrel} onChange={handle} />
    </div>
  );
}

const TT_OPTS = [
  { value: 1, label: "1 – < 15 phút" }, { value: 2, label: "2 – 15–30 phút" },
  { value: 3, label: "3 – 30–60 phút" }, { value: 4, label: "4 – > 1 giờ" },
];
const ST_OPTS = [
  { value: 1, label: "1 – < 2 giờ" }, { value: 2, label: "2 – 2–5 giờ" },
  { value: 3, label: "3 – 5–10 giờ" }, { value: 4, label: "4 – > 10 giờ" },
];

function Tab2({ form, handle }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "20px" }}>
        <FSelect label="Lý do chọn trường (reason)" name="reason" value={form.reason} onChange={handle}
          options={[{ value: "home", label: "Gần nhà" }, { value: "reputation", label: "Danh tiếng" },
            { value: "course", label: "Môn học" }, { value: "other", label: "Khác" }]} />
        <FSelect label="Thời gian đi học (traveltime)" name="traveltime" value={form.traveltime} onChange={handle} options={TT_OPTS} />
        <FSelect label="Thời gian tự học (studytime)" name="studytime" value={form.studytime} onChange={handle} options={ST_OPTS} />
        <FNumber label="Số lần trượt môn (failures, 0–3)" name="failures" min={0} max={3} value={form.failures} onChange={handle} />
      </div>
      <div style={{ backgroundColor: "#0a1628", borderRadius: "12px", padding: "16px", border: "1px solid #1e3a5f", display: "flex", flexDirection: "column", gap: "10px" }}>
        <p style={{ color: "#64748b", fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", margin: 0 }}>Hỗ trợ & Hoạt động</p>
        <FToggle label="Hỗ trợ từ trường (schoolsup)" name="schoolsup" value={form.schoolsup} onChange={handle} />
        <FToggle label="Hỗ trợ từ gia đình (famsup)" name="famsup" value={form.famsup} onChange={handle} />
        <FToggle label="Học thêm có thu phí (paid)" name="paid" value={form.paid} onChange={handle} />
        <FToggle label="Tham gia ngoại khóa (activities)" name="activities" value={form.activities} onChange={handle} />
        <FToggle label="Mong muốn học đại học / cao học (higher)" name="higher" value={form.higher} onChange={handle} />
      </div>
    </div>
  );
}

function Tab3({ form, handle }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ backgroundColor: "#0a1628", borderRadius: "12px", padding: "16px", border: "1px solid #1e3a5f", display: "flex", flexDirection: "column", gap: "10px" }}>
        <p style={{ color: "#64748b", fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", margin: 0 }}>Lối sống</p>
        <FToggle label="Từng học mẫu giáo (nursery)" name="nursery" value={form.nursery} onChange={handle} />
        <FToggle label="Có internet ở nhà (internet)" name="internet" value={form.internet} onChange={handle} />
        <FToggle label="Đang có mối quan hệ tình cảm (romantic)" name="romantic" value={form.romantic} onChange={handle} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "20px" }}>
        <FSlider label="Thời gian rảnh rỗi (freetime)" name="freetime" min={1} max={5} value={form.freetime} onChange={handle} />
        <FSlider label="Tần suất đi chơi (goout)" name="goout" min={1} max={5} value={form.goout} onChange={handle} />
        <FSlider label="Uống rượu ngày thường (Dalc)" name="Dalc" min={1} max={5} value={form.Dalc} onChange={handle} />
        <FSlider label="Uống rượu cuối tuần (Walc)" name="Walc" min={1} max={5} value={form.Walc} onChange={handle} />
        <FSlider label="Sức khỏe hiện tại (health)" name="health" min={1} max={5} value={form.health} onChange={handle} />
        <FNumber label="Số buổi nghỉ học (absences, 0–93)" name="absences" min={0} max={93} value={form.absences} onChange={handle} />
      </div>
      <div style={{ backgroundColor: "#0a1628", borderRadius: "12px", padding: "16px", border: "1px solid rgba(99,102,241,0.3)", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px", alignItems: "end" }}>
        <div style={{ gridColumn: "1 / -1" }}>
          <p style={{ color: "#818cf8", fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", margin: "0 0 8px 0" }}>📊 Điểm số kỳ trước</p>
        </div>
        <FNumber label="Điểm kỳ 1 (G1, 0–20)" name="G1" min={0} max={20} value={form.G1} onChange={handle} />
        <FNumber label="Điểm kỳ 2 (G2, 0–20)" name="G2" min={0} max={20} value={form.G2} onChange={handle} />
        <div style={{ paddingBottom: "8px" }}>
          <p style={{ color: "#64748b", fontSize: "12px", margin: "0 0 2px 0" }}>Dự đoán G3</p>
          <p style={{ color: "#818cf8", fontSize: "28px", fontWeight: 700, fontFamily: "monospace", margin: 0 }}>?</p>
        </div>
      </div>
    </div>
  );
}

// ── Result components ─────────────────────────────────────────────────────────
function SingleResult({ result }) {
  return (
    <div style={{ backgroundColor: "#0a1628", borderRadius: "16px", border: "1px solid #1e3a5f", padding: "24px", display: "flex", flexDirection: "column", alignItems: "center", gap: "16px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#94a3b8", fontSize: "13px" }}>
        <Brain size={15} color="#818cf8" />
        <span>{result.model_name || result.display_name || "Model"}</span>
      </div>
      <CircularGauge score={result.predicted_G3} />
      {result.message && <p style={{ textAlign: "center", color: "#94a3b8", fontSize: "13px", maxWidth: "260px", lineHeight: 1.5 }}>{result.message}</p>}
    </div>
  );
}

function AllResults({ results }) {
  const entries = Object.entries(results);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <p style={{ color: "#64748b", fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", margin: 0 }}>So sánh tất cả mô hình</p>
      {entries.map(([model, res]) => {
        const score = res?.predicted_G3 ?? res?.prediction ?? null;
        if (score === null) return null;
        const color = gradeColor(score);
        return (
          <div key={model} style={{ backgroundColor: "#0a1628", borderRadius: "12px", border: "1px solid #1e3a5f", padding: "12px 16px", display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ flexShrink: 0, width: "44px", height: "44px", borderRadius: "50%", backgroundColor: color + "22", color, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "monospace", fontWeight: 700, fontSize: "16px" }}>
              {score.toFixed(1)}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ color: "#e2e8f0", fontSize: "13px", fontWeight: 500, margin: "0 0 6px 0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {res?.display_name || model}
              </p>
              <div style={{ height: "6px", backgroundColor: "#1e3a5f", borderRadius: "999px", overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${(score / 20) * 100}%`, backgroundColor: color, borderRadius: "999px", transition: "width 0.8s ease" }} />
              </div>
            </div>
            <span style={{ flexShrink: 0, backgroundColor: color + "22", color, fontSize: "11px", padding: "2px 8px", borderRadius: "999px", fontWeight: 600 }}>
              {gradeLabel(score)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
const TABS = [
  { id: 0, label: "Cá nhân & Gia đình", icon: User },
  { id: 1, label: "Học tập & Bối cảnh", icon: BookOpen },
  { id: 2, label: "Lối sống & Điểm số", icon: Heart },
];

export default function App() {
  const [tab, setTab] = useState(0);
  const [form, setForm] = useState(INIT_FORM);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("__all__");
  const [result, setResult] = useState(null);
  const [allResults, setAllResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${BASE_URL}/models`);
        if (!r.ok) throw new Error();
        setModels(await r.json());
      } catch { setModels([]); }
      finally { setModelsLoading(false); }
    })();
  }, []);

  const handle = (e) => {
    const { name, value } = e.target;
    setForm(f => ({ ...f, [name]: NUM_FIELDS.includes(name) ? Number(value) : value }));
  };

  const fillMock = () => { setForm(MOCK_DATA); setResult(null); setAllResults(null); setError(null); };
  const reset = () => { setForm(INIT_FORM); setResult(null); setAllResults(null); setError(null); };

  const submit = async () => {
    setLoading(true); setError(null); setResult(null); setAllResults(null);
    try {
      const body = JSON.stringify(form);
      const headers = { "Content-Type": "application/json" };
      if (selectedModel === "__all__") {
        const r = await fetch(`${BASE_URL}/predict/all`, { method: "POST", headers, body });
        if (!r.ok) { const d = await r.json(); throw new Error(d.detail || "Lỗi API"); }
        setAllResults(await r.json());
      } else {
        const r = await fetch(`${BASE_URL}/predict/${selectedModel}`, { method: "POST", headers, body });
        if (!r.ok) { const d = await r.json(); throw new Error(d.detail || "Lỗi API"); }
        setResult(await r.json());
      }
    } catch (e) {
      setError(e.message || `Không kết nối được tới server tại ${BASE_URL}`);
    } finally { setLoading(false); }
  };

  const hasResult = result || allResults;

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#060e1a", color: "#fff", fontFamily: "system-ui, -apple-system, sans-serif" }}>
      {/* Header */}
      <header style={{ borderBottom: "1px solid #1e3a5f", backgroundColor: "#080f1f", padding: "12px 16px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: "36px", height: "36px", borderRadius: "10px", backgroundColor: "#4f46e5", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <School size={18} color="#fff" />
          </div>
          <div>
          <h1 style={{ margin: 0, fontSize: "16px", fontWeight: 700, lineHeight: 1.2, color: "#ffffff" }}>Hệ thống Dự đoán Kết quả Học tập</h1>
            <p style={{ margin: 0, fontSize: "12px", color: "#64748b", lineHeight: 1.2 }}>Dự đoán điểm G3 cuối kỳ dựa trên 32 đặc trưng của sinh viên</p>
          </div>
        </div>
      </header>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "24px 16px", display: "flex", gap: "24px", flexWrap: "wrap" }}>
        {/* ── LEFT: Form ─── */}
        <div style={{ flex: "1 1 500px", minWidth: 0, display: "flex", flexDirection: "column", gap: "16px" }}>

          {/* Model selector */}
          <div style={{ backgroundColor: "#0a1628", borderRadius: "16px", border: "1px solid #1e3a5f", padding: "14px 16px", display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#94a3b8", flexShrink: 0 }}>
              <BarChart2 size={15} color="#818cf8" />
              <span style={{ fontSize: "13px", fontWeight: 500 }}>Mô hình dự đoán</span>
            </div>
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              disabled={modelsLoading}
              style={{ flex: 1, minWidth: "180px", backgroundColor: "#0f1e35", border: "1px solid #1e3a5f", color: "#ffffff", borderRadius: "8px", padding: "7px 10px", fontSize: "13px", outline: "none", cursor: "pointer" }}
            >
              <option value="__all__" style={{ backgroundColor: "#0f1e35", color: "#fff" }}>⚡ Dự đoán bằng tất cả mô hình</option>
              {models.map(m => (
                <option key={m.model_name} value={m.model_name} style={{ backgroundColor: "#0f1e35", color: "#fff" }}>
                  {m.display_name || m.model_name}
                </option>
              ))}
            </select>
            {modelsLoading && <Loader2 size={15} color="#64748b" className="animate-spin" />}
          </div>

          {/* Tab panel */}
          <div style={{ backgroundColor: "#0a1628", borderRadius: "16px", border: "1px solid #1e3a5f", overflow: "hidden" }}>
            {/* Tab nav */}
            <div style={{ display: "flex", borderBottom: "1px solid #1e3a5f" }}>
              {TABS.map(t => {
                const Icon = t.icon;
                const active = tab === t.id;
                return (
                  <button key={t.id} onClick={() => setTab(t.id)} style={{
                    flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: "6px",
                    padding: "12px 8px", fontSize: "12px", fontWeight: 500, cursor: "pointer",
                    borderBottom: active ? "2px solid #6366f1" : "2px solid transparent",
                    color: active ? "#a5b4fc" : "#64748b",
                    backgroundColor: active ? "rgba(99,102,241,0.05)" : "transparent",
                    border: "none", borderBottom: active ? "2px solid #6366f1" : "2px solid transparent",
                    transition: "all 0.15s",
                  }}>
                    <Icon size={13} />
                    <span style={{ display: "inline" }}>{t.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Tab body */}
            <div style={{ padding: "20px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "16px" }}>
                <span style={{ width: "20px", height: "20px", borderRadius: "50%", backgroundColor: "rgba(99,102,241,0.2)", color: "#a5b4fc", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "11px", fontWeight: 700, flexShrink: 0 }}>{tab + 1}</span>
                <span style={{ color: "#64748b", fontSize: "12px" }}>{TABS[tab].label}</span>
              </div>
              {tab === 0 && <Tab1 form={form} handle={handle} />}
              {tab === 1 && <Tab2 form={form} handle={handle} />}
              {tab === 2 && <Tab3 form={form} handle={handle} />}
            </div>

            {/* Tab footer */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 20px 16px" }}>
              <button onClick={() => setTab(t => Math.max(0, t - 1))} disabled={tab === 0}
                style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "13px", color: tab === 0 ? "#334155" : "#94a3b8", background: "none", border: "none", cursor: tab === 0 ? "default" : "pointer" }}>
                <ChevronLeft size={15} /> Trước
              </button>
              <div style={{ display: "flex", gap: "6px" }}>
                {TABS.map(t => (
                  <button key={t.id} onClick={() => setTab(t.id)}
                    style={{ width: "8px", height: "8px", borderRadius: "50%", border: "none", cursor: "pointer",
                      backgroundColor: tab === t.id ? "#6366f1" : "#1e3a5f", transition: "background-color 0.2s" }} />
                ))}
              </div>
              <button onClick={() => setTab(t => Math.min(2, t + 1))} disabled={tab === 2}
                style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "13px", color: tab === 2 ? "#334155" : "#94a3b8", background: "none", border: "none", cursor: tab === 2 ? "default" : "pointer" }}>
                Tiếp <ChevronRight size={15} />
              </button>
            </div>
          </div>

          {/* Action buttons */}
          <div style={{ display: "flex", gap: "10px" }}>
            <button onClick={fillMock} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "10px 16px", borderRadius: "10px", border: "1px solid #1e3a5f", backgroundColor: "transparent", color: "#94a3b8", fontSize: "13px", cursor: "pointer" }}>
              <FlaskConical size={14} /> Dữ liệu mẫu
            </button>
            <button onClick={reset} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "10px 16px", borderRadius: "10px", border: "1px solid #1e3a5f", backgroundColor: "transparent", color: "#94a3b8", fontSize: "13px", cursor: "pointer" }}>
              <RefreshCw size={14} /> Đặt lại
            </button>
            <button onClick={submit} disabled={loading} style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", padding: "10px 20px", borderRadius: "10px", border: "none", backgroundColor: loading ? "#3730a3" : "#4f46e5", color: "#fff", fontSize: "14px", fontWeight: 600, cursor: loading ? "not-allowed" : "pointer", transition: "background-color 0.2s" }}>
              {loading ? <><Loader2 size={16} className="animate-spin" /> Đang xử lý...</> : <><Brain size={16} /> Dự đoán G3</>}
            </button>
          </div>
        </div>

        {/* ── RIGHT: Results ─── */}
        <div style={{ width: "320px", flexShrink: 0, display: "flex", flexDirection: "column", gap: "14px" }}>

          {error && (
            <div style={{ backgroundColor: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.35)", borderRadius: "14px", padding: "14px", display: "flex", gap: "10px" }}>
              <AlertCircle size={17} color="#f87171" style={{ flexShrink: 0, marginTop: "2px" }} />
              <div>
                <p style={{ margin: "0 0 4px 0", fontSize: "13px", fontWeight: 600, color: "#fca5a5" }}>Lỗi kết nối</p>
                <p style={{ margin: 0, fontSize: "12px", color: "rgba(252,165,165,0.75)", lineHeight: 1.5 }}>{error}</p>
              </div>
            </div>
          )}

          {!hasResult && !error && !loading && (
            <div style={{ backgroundColor: "#0a1628", borderRadius: "16px", border: "1px dashed #1e3a5f", padding: "40px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: "12px", textAlign: "center" }}>
              <div style={{ width: "48px", height: "48px", borderRadius: "50%", backgroundColor: "rgba(99,102,241,0.1)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Brain size={22} color="#818cf8" />
              </div>
              <p style={{ color: "#64748b", fontSize: "13px", lineHeight: 1.6, margin: 0 }}>
                Kết quả dự đoán sẽ hiển thị ở đây sau khi bạn nhấn <strong style={{ color: "#94a3b8" }}>Dự đoán G3</strong>.
              </p>
            </div>
          )}

          {loading && (
            <div style={{ backgroundColor: "#0a1628", borderRadius: "16px", border: "1px solid #1e3a5f", padding: "32px", display: "flex", flexDirection: "column", alignItems: "center", gap: "16px" }}>
              <div style={{ width: "56px", height: "56px", borderRadius: "50%", border: "4px solid #1e3a5f", borderTopColor: "#6366f1", animation: "spin 0.8s linear infinite" }} />
              <p style={{ color: "#64748b", fontSize: "13px", margin: 0 }}>Đang tính toán...</p>
            </div>
          )}

          {result && !loading && <SingleResult result={result} />}
          {allResults && !loading && (
            <div style={{ backgroundColor: "#0a1628", borderRadius: "16px", border: "1px solid #1e3a5f", padding: "16px" }}>
              <AllResults results={allResults} />
            </div>
          )}

          {hasResult && !loading && (
            <div style={{ backgroundColor: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.25)", borderRadius: "10px", padding: "10px 12px", display: "flex", alignItems: "center", gap: "8px" }}>
              <CheckCircle2 size={14} color="#4ade80" />
              <p style={{ margin: 0, fontSize: "12px", color: "#4ade80" }}>Dự đoán hoàn tất. Thay đổi thông tin và chạy lại bất kỳ lúc nào.</p>
            </div>
          )}

          {/* Tips */}
          <div style={{ backgroundColor: "#0a1628", borderRadius: "16px", border: "1px solid #1e3a5f", padding: "14px 16px" }}>
            <p style={{ margin: "0 0 8px 0", color: "#64748b", fontSize: "11px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>Hướng dẫn nhanh</p>
            {["Điền đầy đủ 3 tab để đảm bảo độ chính xác", 'Nhấn "Dữ liệu mẫu" để test nhanh toàn bộ form', 'Chọn "Tất cả mô hình" để so sánh kết quả', "G3 được tính theo thang điểm 0 – 20"].map((tip, i) => (
              <p key={i} style={{ margin: "4px 0 0 0", color: "#475569", fontSize: "12px", lineHeight: 1.5 }}>• {tip}</p>
            ))}
          </div>
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
