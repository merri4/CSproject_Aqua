import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Bot,
  Check,
  Database,
  FileText,
  Moon,
  PlayCircle,
  RefreshCcw,
  Send,
  Sun,
  Video,
  X,
  Zap
} from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const STREAM_URL = import.meta.env.VITE_OMNIVERSE_STREAM_URL || 'http://localhost:8011/streaming/webrtc-client';

function classNames(...xs) {
  return xs.filter(Boolean).join(' ');
}

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function formatTime(v) {
  if (!v) return '';
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v).slice(11, 19);
  return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function latestValue(rows, key) {
  if (!rows.length) return '-';
  const val = rows[rows.length - 1]?.[key];
  return typeof val === 'number' ? val.toFixed(2) : val ?? '-';
}

function riskBadge(level) {
  const map = {
    normal: 'badge normal',
    watch: 'badge watch',
    warning: 'badge warning',
    critical: 'badge critical'
  };
  return map[level] || 'badge watch';
}

export default function App() {
  const [dark, setDark] = useState(true);
  const [rows, setRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [selectedSeries, setSelectedSeries] = useState([]);
  const [loadingSensors, setLoadingSensors] = useState(false);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [proposal, setProposal] = useState(null);
  const [proposalLoading, setProposalLoading] = useState(false);
  const [controlResult, setControlResult] = useState(null);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '현재 센서 상태를 기반으로 질문하거나, AI 조치 제안을 요청하세요.' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [report, setReport] = useState('');
  const [reportLoading, setReportLoading] = useState(false);
  const chatEndRef = useRef(null);

  const timeColumn = columns[0] || 'timestamp';
  const sensorColumns = useMemo(() => columns.slice(1), [columns]);

  const chartData = useMemo(() => {
    return rows.map((r) => ({
      ...r,
      _time: formatTime(r[timeColumn])
    }));
  }, [rows, timeColumn]);

  async function loadStatus() {
    try {
      const data = await api('/api/status');
      setServiceStatus(data);
    } catch (err) {
      setServiceStatus({ error: err.message });
    }
  }

  async function loadSensors() {
    setLoadingSensors(true);
    try {
      const data = await api('/api/sensors/recent?hours=1&limit=7200');
      setRows(data.rows || []);
      setColumns(data.columns || []);
      if (!selectedSeries.length && data.columns?.length > 1) {
        setSelectedSeries(data.columns.slice(1, 4));
      }
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `센서 로딩 오류: ${err.message}` }]);
    } finally {
      setLoadingSensors(false);
    }
  }

  useEffect(() => {
    loadSensors();
    loadStatus();
    const sensorId = setInterval(loadSensors, 5000);
    const statusId = setInterval(loadStatus, 15000);
    return () => {
      clearInterval(sensorId);
      clearInterval(statusId);
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  }, [dark]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function requestProposal() {
    setProposalLoading(true);
    setControlResult(null);
    try {
      const p = await api('/api/ai/propose', { method: 'POST', body: JSON.stringify({}) });
      setProposal(p);
    } catch (err) {
      setProposal({ summary: `AI 제안 생성 오류: ${err.message}`, risk_level: 'watch', actions: [] });
    } finally {
      setProposalLoading(false);
    }
  }

  async function decide(decision) {
    if (!proposal) return;
    try {
      const result = await api('/api/ai/decision', {
        method: 'POST',
        body: JSON.stringify({ proposal, decision })
      });
      setControlResult(result);
      if (decision === 'confirm') {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `조치가 confirm되었습니다. Omniverse controller로 보낼 payload가 생성되었습니다.\n${JSON.stringify(result.control_payload, null, 2)}` }
        ]);
      }
    } catch (err) {
      setControlResult({ status: 'error', message: err.message });
    }
  }

  async function sendChat() {
    const message = chatInput.trim();
    if (!message) return;
    setChatInput('');
    setChatLoading(true);
    setMessages((prev) => [...prev, { role: 'user', content: message }]);
    try {
      const res = await api('/api/chat', { method: 'POST', body: JSON.stringify({ message }) });
      setMessages((prev) => [...prev, { role: 'assistant', content: res.reply }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `채팅 오류: ${err.message}` }]);
    } finally {
      setChatLoading(false);
    }
  }

  async function generateOpsReport() {
    setReportLoading(true);
    try {
      const res = await api('/api/report', {
        method: 'POST',
        body: JSON.stringify({ operator_note: '대시보드에서 수동 생성한 운영 리포트' })
      });
      setReport(res.report);
      if (res.proposal) setProposal(res.proposal);
    } catch (err) {
      setReport(`리포트 생성 오류: ${err.message}`);
    } finally {
      setReportLoading(false);
    }
  }

  function toggleSeries(c) {
    setSelectedSeries((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));
  }

  function connectionClass(ok) {
    if (ok === true) return 'connection ok';
    if (ok === false) return 'connection error';
    return 'connection watch';
  }

  const connections = [
    { label: 'DB', ok: serviceStatus?.db?.ok, detail: serviceStatus?.db?.table || serviceStatus?.db?.error || 'checking' },
    { label: 'LLM', ok: serviceStatus?.llm?.ok, detail: serviceStatus?.llm?.model || serviceStatus?.llm?.error || 'checking' },
    { label: 'RAG', ok: serviceStatus?.rag?.ok, detail: serviceStatus?.rag?.mode || serviceStatus?.rag?.warning || 'checking' },
    { label: 'WebRTC', ok: Boolean(STREAM_URL), detail: STREAM_URL }
  ];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Closed-loop Digital Twin</div>
          <h1>Salmon Farm AI Operations Dashboard</h1>
        </div>
        <div className="topbar-actions">
          <button className="ghost" onClick={loadSensors} disabled={loadingSensors}>
            <RefreshCcw size={16} /> Refresh
          </button>
          <button className="ghost" onClick={() => setDark((v) => !v)}>
            {dark ? <Sun size={16} /> : <Moon size={16} />} {dark ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="card hero-stream">
          <div className="card-header">
            <div>
              <span className="section-label"><Video size={16} /> Omniverse WebRTC</span>
              <h2>Live Twin View</h2>
            </div>
            <span className="live-dot">LIVE</span>
          </div>
          <div className="stream-frame">
            <iframe title="Omniverse WebRTC Stream" src={STREAM_URL} allow="camera; microphone; fullscreen; autoplay" />
          </div>
          <p className="muted small">VITE_OMNIVERSE_STREAM_URL 값에 Omniverse Streaming Client URL을 넣으면 이 영역에 표시됩니다.</p>
        </section>

        <section className="card status-card">
          <div className="card-header">
            <div>
              <span className="section-label"><Activity size={16} /> Current State</span>
              <h2>Latest Sensor Snapshot</h2>
            </div>
          </div>
          <div className="connection-grid">
            {connections.map((item) => (
              <div className={connectionClass(item.ok)} key={item.label}>
                <span>{item.label}</span>
                <strong>{item.ok === true ? 'connected' : item.ok === false ? 'error' : 'checking'}</strong>
                <small title={item.detail}>{item.detail}</small>
              </div>
            ))}
          </div>
          <div className="metric-grid">
            {sensorColumns.map((c) => (
              <div className="metric" key={c}>
                <div className="metric-key">{c}</div>
                <div className="metric-value">{latestValue(rows, c)}</div>
              </div>
            ))}
          </div>
          <button className="primary full" onClick={requestProposal} disabled={proposalLoading}>
            <Zap size={16} /> {proposalLoading ? 'AI 분석 중...' : 'AI 조치 제안 받기'}
          </button>
          {proposal && (
            <div className="proposal-box">
              <div className="proposal-top">
                <span className={riskBadge(proposal.risk_level)}>{proposal.risk_level}</span>
              </div>
              <p>{proposal.summary}</p>
              <div className="action-list">
                {(proposal.actions || []).length === 0 && <div className="muted">제안된 제어 조치가 없습니다.</div>}
                {(proposal.actions || []).map((a, idx) => (
                  <div className="action-item" key={`${a.variable}-${idx}`}>
                    <strong>{a.variable}</strong>
                    <span>{a.direction} {a.amount}{a.unit ? ` ${a.unit}` : ''}</span>
                    <small>{a.rationale}</small>
                  </div>
                ))}
              </div>
              <div className="decision-row">
                <button className="confirm" onClick={() => decide('confirm')}><Check size={16} /> Confirm</button>
                <button className="decline" onClick={() => decide('decline')}><X size={16} /> Decline</button>
              </div>
            </div>
          )}
          {controlResult && <pre className="result-box">{JSON.stringify(controlResult, null, 2)}</pre>}
        </section>

        <section className="card chart-card">
          <div className="card-header">
            <div>
              <span className="section-label"><Database size={16} /> Last 1 Hour</span>
              <h2>Raw Sensor Trends</h2>
            </div>
          </div>
          <div className="series-pills">
            {sensorColumns.map((c) => (
              <button
                key={c}
                onClick={() => toggleSeries(c)}
                className={classNames('pill', selectedSeries.includes(c) && 'active')}
              >
                {c}
              </button>
            ))}
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 16, right: 20, left: -20, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="_time" minTickGap={48} />
                <YAxis />
                <Tooltip />
                {selectedSeries.map((c) => (
                  <Line key={c} type="monotone" dataKey={c} dot={false} strokeWidth={2} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card raw-card">
          <div className="card-header">
            <div>
              <span className="section-label"><Database size={16} /> SQL Result</span>
              <h2>Recent Sensor Values</h2>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {rows.slice(-18).reverse().map((r, i) => (
                  <tr key={i}>
                    {columns.map((c) => <td key={c}>{c === timeColumn ? formatTime(r[c]) : String(r[c] ?? '')}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card chat-card">
          <div className="card-header">
            <div>
              <span className="section-label"><Bot size={16} /> LLM + RAG</span>
              <h2>Operator Chat</h2>
            </div>
          </div>
          <div className="chat-log">
            {messages.map((m, idx) => (
              <div key={idx} className={classNames('msg', m.role)}>
                <div className="msg-role">{m.role === 'user' ? 'Operator' : 'AI Expert'}</div>
                <div className="msg-body">{m.content}</div>
              </div>
            ))}
            {chatLoading && <div className="msg assistant"><div className="msg-role">AI Expert</div><div className="msg-body">응답 생성 중...</div></div>}
            <div ref={chatEndRef} />
          </div>
          <div className="chat-input">
            <input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') sendChat(); }}
              placeholder="예: 현재 수온과 DO 기준으로 위험도를 평가해줘"
            />
            <button className="primary" onClick={sendChat}><Send size={16} /></button>
          </div>
        </section>

        <section className="card report-card">
          <div className="card-header">
            <div>
              <span className="section-label"><FileText size={16} /> AI Report</span>
              <h2>Operations Report</h2>
            </div>
            <button className="primary" onClick={generateOpsReport} disabled={reportLoading}>
              <PlayCircle size={16} /> {reportLoading ? '생성 중...' : '리포트 생성'}
            </button>
          </div>
          <div className="report-body">
            {report ? <pre>{report}</pre> : <p className="muted">최근 1시간 센서 데이터, RAG 지식, AI 조치 제안을 종합해 운영 리포트를 생성합니다.</p>}
          </div>
        </section>
      </main>
    </div>
  );
}
