// In local development Vite proxies /api to the backend. VITE_API_BASE remains
// available for deployments where the API is hosted on a different origin.
const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";

export async function streamAsk(id, question, chat_history = [], provider = "gemini-lite", onDelta = () => {}) {
  if (USE_MOCK) { const result = await api.ask(id, question, chat_history, provider); onDelta(result.answer); return result; }
  const response = await fetch(`${API_BASE}/api/analysis/${encodeURIComponent(id)}/ask/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, chat_history: chat_history.slice(-6), provider }) });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let result = null;
  while (true) { const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); const events = buffer.split("\n\n"); buffer = events.pop() || ""; for (const event of events) { const line = event.split("\n").find(item => item.startsWith("data: ")); if (!line) continue; const payload = JSON.parse(line.slice(6)); if (payload.type === "delta") onDelta(payload.text); if (payload.type === "done") result = payload.result; } }
  if (!result) throw new Error("stream ended without a result"); return result;
}

async function mock(name) {
  const response = await fetch(`/${name}.json`);
  if (!response.ok) throw new Error(`mock ${name} unavailable`);
  return response.json();
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

export const api = {
  documents: () => USE_MOCK ? Promise.resolve([{ document_id: "mock-001", title: "행정안전부 RFP 2026 지능형민원시스템 구축", organization: "행정안전부", difficulty: "high", status: "ready" }]) : request("/api/documents"),
  page: (id, page) => USE_MOCK ? Promise.resolve({ document_id: id, page, page_count: 60, text: "Mock RFP 원문 페이지입니다. 실제 API 모드에서는 PDF 추출 본문이 표시됩니다.", headings: [], image_count: 0, ocr_applied: false }) : request(`/api/document?document_id=${encodeURIComponent(id)}&page=${page}`),
  toc: id => USE_MOCK ? Promise.resolve({ document_id: id, items: [{ id: "mock-toc-1", title: "입찰 참가자격", page: 11 }, { id: "mock-toc-2", title: "제출서류", page: 31 }, { id: "mock-toc-3", title: "요구사항", page: 15 }] }) : request(`/api/toc?document_id=${encodeURIComponent(id)}`),
  searchPages: (id, q) => USE_MOCK ? Promise.resolve({ document_id: id, results: [{ page: 31, excerpt: `Mock 검색 결과: ${q} 관련 제출서류 내용입니다.`, score: 1 }] }) : request(`/api/search?document_id=${encodeURIComponent(id)}&q=${encodeURIComponent(q)}`),
  health: () => request("/api/health"),
  evaluationSummary: () => request("/api/evaluation/summary"),
  upload: (file, title = "") => request(`/api/documents/upload?filename=${encodeURIComponent(file.name)}&title=${encodeURIComponent(title)}`, { method: "PUT", headers: { "Content-Type": "application/pdf" }, body: file }),
  overview: id => USE_MOCK ? mock("overview") : request(`/api/analysis/${encodeURIComponent(id)}/overview`),
  risks: id => USE_MOCK ? mock("risks") : request(`/api/analysis/${encodeURIComponent(id)}/risks`),
  eligibility: id => USE_MOCK ? mock("eligibility") : request(`/api/analysis/${encodeURIComponent(id)}/eligibility`),
  deliverables: id => USE_MOCK ? mock("deliverables") : request(`/api/analysis/${encodeURIComponent(id)}/deliverables`),
  requirements: id => USE_MOCK ? mock("requirements") : request(`/api/analysis/${encodeURIComponent(id)}/requirements`),
  ask: (id, question, chat_history = [], provider = "gemini-lite") => USE_MOCK ? Promise.resolve({ question, answer: `Mock 답변: ${question}에 대한 문서 근거 요약입니다.`, is_answerable: true, caveat: null, citations: [{ source_id: "S1", chunk_id: "mock-001", document_id: id, document_name: "행정안전부 RFP 2026 지능형민원시스템 구축", page_start: 31, page_end: 31, requirement_ids: [] }], retrieved_chunk_ids: ["mock-001"], retriever: "mock", model: provider, search_latency_ms: 1, generation_latency_ms: 1 }) : request(`/api/analysis/${encodeURIComponent(id)}/ask`, { method: "POST", body: JSON.stringify({ question, chat_history: chat_history.slice(-6), provider }) }),
  askStream: (id, question, chat_history = [], provider = "gemini-lite") => request(`/api/analysis/${encodeURIComponent(id)}/ask/stream`, { method: "POST", body: JSON.stringify({ question, chat_history: chat_history.slice(-6), provider }) }),
  updateEligibility: (id, itemId, user_status) => USE_MOCK ? Promise.resolve({ item_id: itemId, user_status }) : request(`/api/state/${encodeURIComponent(id)}/eligibility/${encodeURIComponent(itemId)}`, { method: "PATCH", body: JSON.stringify({ user_status }) }),
  updateRisk: (id, itemId, user_status) => USE_MOCK ? Promise.resolve({ item_id: itemId, user_status }) : request(`/api/state/${encodeURIComponent(id)}/risk/${encodeURIComponent(itemId)}`, { method: "PATCH", body: JSON.stringify({ user_status }) }),
  updateDeliverable: (id, itemId, values) => USE_MOCK ? Promise.resolve({ item_id: itemId, ...values }) : request(`/api/state/${encodeURIComponent(id)}/deliverable/${encodeURIComponent(itemId)}`, { method: "PATCH", body: JSON.stringify(values) }),
};
