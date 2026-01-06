import { useState } from "react";
import "./App.css";
import student from "./assets/student.svg";

function Counter({ label, name, min, max, step = 1, value, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="counter">
        <button type="button" onClick={() => onChange(name, Math.max(min, value - step))}>−</button>
        <span>{value}</span>
        <button type="button" onClick={() => onChange(name, Math.min(max, value + step))}>+</button>
      </div>
    </div>
  );
}

function App() {
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = "https://vishaka.onrender.com/predict";

  const updateValue = (key, value) => {
    setFormData({ ...formData, [key]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card" style={{marginLeft: '100px'}}>
        <div className="card-left" >
        <h1>🎓  Student Performance Predictor (MAP with CBT)</h1>
        <p className="subtitle">
          Fill only what you know — the model handles the rest ✨
        </p>

        <form onSubmit={handleSubmit}>
          <div className="grid">
            <Counter label="Study Hours / Day" name="study_hours_per_day" min={0} max={12} value={formData.study_hours_per_day || 5} onChange={updateValue} />
            <Counter label="Sleep Hours" name="sleep_hours" min={0} max={12} value={formData.sleep_hours || 7} onChange={updateValue} />
            <Counter label="Attendance %" name="attendance_percentage" min={0} max={100} value={formData.attendance_percentage || 85} onChange={updateValue} />
            <Counter label="Motivation Level" name="motivation_level" min={0} max={10} value={formData.motivation_level || 7} onChange={updateValue} />
            <Counter label="Stress Level" name="stress_level" min={0} max={10} value={formData.stress_level || 4} onChange={updateValue} />
            <Counter label="Screen Time (hrs)" name="screen_time" min={0} max={12} value={formData.screen_time || 5} onChange={updateValue} />
          </div>

          <div className="field">
            <label>Learning Style</label>
            <select onChange={(e) => updateValue("learning_style", e.target.value)}>
              <option value="">Select</option>
              <option>Visual</option>
              <option>Auditory</option>
              <option>Reading</option>
              <option>Kinesthetic</option>
            </select>
          </div>

          <div className="field">
            <label>Study Environment</label>
            <select onChange={(e) => updateValue("study_environment", e.target.value)}>
              <option value="">Select</option>
              <option>Quiet</option>
              <option>Moderate</option>
              <option>Noisy</option>
            </select>
          </div>

          <button className="predict-btn" type="submit">
            {loading ? "Predicting..." : "Predict Score"}
          </button>
        </form>

        {result && (
          <div className="result">
            <h2>📊 Prediction</h2>
            <p><strong>Exam Score:</strong> {result.predicted_exam_score}</p>
            <p><strong>Performance:</strong> {result.predicted_performance_class}</p>
          </div>
        )}
        </div>
       
      </div>
      <img src={'./student.png'} alt="student" className="student-img" style={{float: 'right', width: '800px'}} />
    </div>
  );
}

export default App;
