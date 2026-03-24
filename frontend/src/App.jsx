import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [summary, setSummary] = useState(null);
  const [hourData, setHourData] = useState([]);
  const [countryData, setCountryData] = useState([]);
  const [mccData, setMccData] = useState([]);
  const [productData, setProductData] = useState([]);
  const [productAmountData, setProductAmountData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [anomalyAmount, setAnomalyAmount] = useState(105);
  const [anomalyHour, setAnomalyHour] = useState(10);
  const [anomalyResult, setAnomalyResult] = useState(null);
  const [anomalyLoading, setAnomalyLoading] = useState(false);
  const [anomalyError, setAnomalyError] = useState("");

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    setLoading(true);
    setError("");

    try {
      const [
        summaryRes,
        hourRes,
        countryRes,
        mccRes,
        productRes,
        productAmountRes,
      ] = await Promise.all([
        fetch(`${API_BASE}/dashboard/summary`),
        fetch(`${API_BASE}/dashboard/hour-distribution`),
        fetch(`${API_BASE}/dashboard/country-distribution`),
        fetch(`${API_BASE}/dashboard/mcc-distribution`),
        fetch(`${API_BASE}/dashboard/product-distribution`),
        fetch(`${API_BASE}/dashboard/product-amount`),
      ]);

      if (
        !summaryRes.ok ||
        !hourRes.ok ||
        !countryRes.ok ||
        !mccRes.ok ||
        !productRes.ok ||
        !productAmountRes.ok
      ) {
        throw new Error("Failed to load dashboard data");
      }

      const summaryJson = await summaryRes.json();
      const hourJson = await hourRes.json();
      const countryJson = await countryRes.json();
      const mccJson = await mccRes.json();
      const productJson = await productRes.json();
      const productAmountJson = await productAmountRes.json();

      setSummary(summaryJson);
      setHourData(hourJson);
      setCountryData(countryJson.slice(0, 10));
      setMccData(mccJson.slice(0, 10));
      setProductData(productJson);
      setProductAmountData(productAmountJson);
    } catch (err) {
      setError(err.message || "Unable to load dashboard");
    } finally {
      setLoading(false);
    }
  };

  const checkAnomaly = async () => {
    setAnomalyLoading(true);
    setAnomalyError("");

    try {
      const res = await fetch(`${API_BASE}/predict-anomaly`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: anomalyAmount,
          hour: anomalyHour,
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to check anomaly");
      }

      const data = await res.json();
      setAnomalyResult(data);
    } catch (err) {
      setAnomalyError(err.message || "Unable to connect");
    } finally {
      setAnomalyLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", padding: "24px" }}>
      <div style={{ maxWidth: "1400px", margin: "0 auto" }}>
        <div style={cardStyle}>
          <h1 style={{ margin: 0, fontSize: "32px" }}>
            Transaction Analytics Dashboard
          </h1>
          <p style={{ marginTop: "8px", color: "#475569" }}>
            Live dashboard from your Excel transaction data
          </p>
          <p style={{ marginTop: "6px", color: "#64748b", fontSize: "14px" }}>
            API Base: {API_BASE}
          </p>
        </div>

        {loading && <div style={cardStyle}>Loading dashboard...</div>}
        {error && <div style={{ ...cardStyle, color: "red" }}>{error}</div>}

        {!loading && !error && summary && (
          <>
            <div style={kpiGridStyle}>
              <KpiCard title="Total Transactions" value={summary.total_transactions} />
              <KpiCard
                title="Total Amount"
                value={Number(summary.total_amount || 0).toLocaleString()}
              />
              <KpiCard
                title="Average Transaction"
                value={Number(summary.average_transaction || 0).toFixed(2)}
              />
            </div>

            <div style={cardStyle}>
              <h3 style={{ marginTop: 0 }}>Anomaly Detection</h3>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr auto",
                  gap: "12px",
                  alignItems: "end",
                }}
              >
                <div>
                  <label>Amount</label>
                  <input
                    type="number"
                    value={anomalyAmount}
                    onChange={(e) => setAnomalyAmount(Number(e.target.value))}
                    style={inputStyle}
                  />
                </div>

                <div>
                  <label>Hour</label>
                  <input
                    type="number"
                    value={anomalyHour}
                    onChange={(e) => setAnomalyHour(Number(e.target.value))}
                    style={inputStyle}
                  />
                </div>

                <button onClick={checkAnomaly} style={buttonStyle}>
                  {anomalyLoading ? "Checking..." : "Check"}
                </button>
              </div>

              {anomalyError && <p style={{ color: "red" }}>{anomalyError}</p>}

              {anomalyResult && (
                <div style={{ marginTop: "16px" }}>
                  <p><strong>Decision:</strong> {anomalyResult.decision}</p>
                  <p><strong>Prediction:</strong> {anomalyResult.anomaly_prediction}</p>
                  <p><strong>Score:</strong> {anomalyResult.anomaly_score}</p>
                </div>
              )}
            </div>

            <div style={chartGridStyle}>
              <ChartCard title="Transactions by Hour">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={hourData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#334155" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Top Countries">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={countryData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="COUNTRY NAME" type="category" width={140} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#0f172a" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Top MCC Categories">
                <ResponsiveContainer width="100%" height={320}>
                  <PieChart>
                    <Pie
                      data={mccData}
                      dataKey="count"
                      nameKey="MCC DESC"
                      outerRadius={100}
                      label
                    >
                      {mccData.map((_, index) => (
                        <Cell
                          key={index}
                          fill={pieColors[index % pieColors.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Transactions by Product">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={productData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="product_sheet" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#1d4ed8" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Amount by Product">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={productAmountData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="product_sheet" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="SETL AMT" fill="#16a34a" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>

            <div style={tableGridStyle}>
              <SimpleTable
                title="Country Distribution"
                rows={countryData}
                nameKey="COUNTRY NAME"
              />

              <SimpleTable
                title="MCC Distribution"
                rows={mccData}
                nameKey="MCC DESC"
              />

              <SimpleTable
                title="Product Distribution"
                rows={productData}
                nameKey="product_sheet"
              />

              <SimpleTable
                title="Amount by Product"
                rows={productAmountData}
                nameKey="product_sheet"
                valueKey="SETL AMT"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function KpiCard({ title, value }) {
  return (
    <div style={cardStyle}>
      <div style={{ color: "#64748b", fontSize: "14px" }}>{title}</div>
      <div style={{ marginTop: "10px", fontSize: "30px", fontWeight: "bold" }}>
        {value}
      </div>
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div style={cardStyle}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <div style={{ height: "320px" }}>{children}</div>
    </div>
  );
}

function SimpleTable({ title, rows, nameKey, valueKey = "count" }) {
  return (
    <div style={cardStyle}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={thStyle}>Category</th>
            <th style={thStyle}>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              <td style={tdStyle}>{row[nameKey]}</td>
              <td style={tdStyle}>
                {typeof row[valueKey] === "number"
                  ? row[valueKey].toLocaleString()
                  : row[valueKey]}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const cardStyle = {
  background: "#ffffff",
  border: "1px solid #e2e8f0",
  borderRadius: "20px",
  padding: "20px",
  boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
  marginBottom: "20px",
};

const kpiGridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
  gap: "20px",
  marginBottom: "20px",
};

const chartGridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))",
  gap: "20px",
  marginBottom: "20px",
};

const tableGridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))",
  gap: "20px",
};

const thStyle = {
  textAlign: "left",
  padding: "10px",
  borderBottom: "1px solid #e2e8f0",
};

const tdStyle = {
  padding: "10px",
  borderBottom: "1px solid #f1f5f9",
};

const inputStyle = {
  width: "100%",
  padding: "10px",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  marginTop: "6px",
};

const buttonStyle = {
  padding: "10px 16px",
  background: "#0f172a",
  color: "#fff",
  border: "none",
  borderRadius: "8px",
  cursor: "pointer",
};

const pieColors = [
  "#0f172a",
  "#334155",
  "#475569",
  "#64748b",
  "#94a3b8",
  "#cbd5e1",
  "#1e293b",
  "#3b82f6",
  "#0ea5e9",
  "#22c55e",
];