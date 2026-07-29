import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, streamAsk } from "./api.js";
import "./style.css";
import "./overview.css";
import "./workspace.css";
import "./chat.css";
import "./search-panel.css";
import "./accessibility.css";
import "./mock-theme.css";
import "./section.css";
import "./evaluation.css";
import "./evaluation-link.css";
import { PipelineEvaluationPage } from "./evaluation_ui.jsx";
import "./evaluation_tabs.css";
import "./requirements.css";

const tabs = ["overview", "go", "deliverables", "requirements", "chat"];
const labels = { overview: "Overview", go: "Go / No-Go", deliverables: "실행 준비", requirements: "요구사항", chat: "AI 질문" };
const cardCriteria = {
  overview: "제출·질의 마감, 참가 자격, 실격·감점·금전 및 계약상 불이익, 제출물과 즉시 조치 항목을 문서에서 추출합니다.",
  go: "참가 자격과 필수 자격 조건, 실격·감점·위약금·계약상 불이익·검토 필요 위험을 기준으로 생성합니다.",
  deliverables: "필수 제출물·산출물·서류를 찾고 형식, 수량, 원본·날인 여부와 마감 정보를 정리합니다.",
  requirements: "기능·성능·보안·운영·인력·산출물·계약 요구사항을 기준으로 분류합니다.",
  chat: "사용자 질문을 요구 항목별로 나누어 검색하고, 확인된 문서 근거만 사용해 답변합니다.",
};
const suggestions = {
  overview: ["제출 준비에 필요한 핵심 항목은?", "마감일과 즉시 할 일을 알려줘"],
  risks: ["실격과 감점·불이익 조건은?", "가장 위험한 항목은 무엇인가?"],
  eligibility: ["참가 자격을 충족하려면 무엇이 필요한가?", "확인하지 못한 자격 조건은?"],
  go: ["참가 자격과 실격 조건은?", "가장 위험한 항목은 무엇인가?"],
  deliverables: ["미완료 제출물과 담당자를 알려줘", "원본·날인 제출물이 무엇인가?"],
  requirements: ["핵심 요구사항을 요약해줘", "보안·성능 요구사항은?"],
  chat: ["실격 조건과 필수 제출물을 알려줘", "이 RFP의 주요 위험을 요약해줘"],
};
let activeDocumentId = "";
function dday(value) {
  if (!value || value.includes("참조") || value.includes("미기재")) return null;
  const days = Math.ceil((new Date(value).getTime() - Date.now()) / 86400000);
  return Number.isFinite(days) ? (days >= 0 ? `D-${days}` : `D+${Math.abs(days)}`) : null;
}

function Evidence({ evidence, pageText, pageCount, onPage }) {
  const [query, setQuery] = useState(""); const [results, setResults] = useState([]);
  const [width, setWidth] = useState(() => Number(localStorage.getItem("rfp-evidence-width")) || 360);
  useEffect(() => { document.documentElement.style.setProperty("--evidence-width", `${width}px`); localStorage.setItem("rfp-evidence-width", String(Math.round(width))); }, [width]);
  const startResize = startEvent => { const layout = startEvent.currentTarget.parentElement; const startX = startEvent.clientX; const startWidth = width; const maxWidth = Math.max(280, layout.getBoundingClientRect().width - 160 - 480 - 8); const move = event => setWidth(Math.min(maxWidth, Math.max(280, startWidth + startX - event.clientX))); const stop = () => { document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", stop); document.body.classList.remove("resizingEvidence"); }; document.body.classList.add("resizingEvidence"); document.addEventListener("pointermove", move); document.addEventListener("pointerup", stop); };
  const resizeBy = delta => setWidth(current => Math.min(Math.max(280, window.innerWidth * .55), Math.max(280, current + delta)));
  const runSearch = async event => { event.preventDefault(); if (!activeDocumentId || !query.trim()) return; try { const response = await api.searchPages(activeDocumentId, query); setResults(response.results || []); } catch { setResults([]); } };
  return <><div className="evidenceResize" role="separator" aria-label="원문 확인 영역 너비 조절" aria-orientation="vertical" tabIndex="0" onPointerDown={startResize} onDoubleClick={() => setWidth(360)} onKeyDown={event => { if (event.key === "ArrowLeft") resizeBy(24); if (event.key === "ArrowRight") resizeBy(-24); }}><span /></div><aside className="evidence"><div className="eyebrow">EVIDENCE / 원문 확인</div><form className="evidenceSearch" onSubmit={runSearch}><input value={query} onChange={e => setQuery(e.target.value)} placeholder="원문 페이지 검색"/><button>검색</button></form>{results.length > 0 && <div className="evidenceSearchResults">{results.map(result => <button key={result.page} onClick={() => { onPage(result.page); setResults([]); setQuery(""); }}><b>p.{result.page}</b> {result.excerpt}</button>)}</div>}{!evidence ? <div className="empty">분석 항목을 선택하면 근거가 표시됩니다.</div> : <><h3>{evidence.document_name}</h3><p>p.{evidence.page_number}{pageCount ? ` / ${pageCount}` : ""} · score {Number(evidence.score || 0).toFixed(2)}</p><div className="pageNav"><button disabled={evidence.page_number <= 1} onClick={() => onPage(evidence.page_number - 1)}>← 이전</button><button disabled={pageCount && evidence.page_number >= pageCount} onClick={() => onPage(evidence.page_number + 1)}>다음 →</button></div><blockquote>{evidence.quote}</blockquote><small>{evidence.requirement_ids?.join(", ") || "요구사항 ID 없음"}</small>{pageText && <div className="pageViewer"><div className="eyebrow">원문 페이지</div><pre>{pageText}</pre></div>}</>}</aside></>;
}

function Toc({ items, onSelect }) {
  const [open, setOpen] = useState(false); const visible = items.slice(0, 80);
  if (!open) return <div className="tocRail"><button className="tocToggle" onClick={() => setOpen(true)} aria-expanded="false" title="문서 목차 열기"><span>☰</span>문서 목차</button></div>;
  return <div className="toc"><div className="tocHead"><div><div className="eyebrow">문서 목차</div><small>{items.length}개{items.length > 80 ? " · 앞 80개 표시" : ""}</small></div><button onClick={() => setOpen(false)} aria-label="문서 목차 닫기">‹</button></div>{visible.length === 0 ? <div className="empty">목차 없음</div> : visible.map(item => <button key={item.id} onClick={() => onSelect(item.page)}>{item.title}<small>p.{item.page}</small></button>)}</div>;
}
function EvidenceButton({ evidence, onEvidence }) { return evidence ? <button className="evidenceLink" onClick={() => onEvidence(evidence)}>{`근거 p.${evidence.page_number}`}</button> : null; }
function riskSummary(items, type) { if (!Array.isArray(items)) return { known: false, total: 0, atRisk: 0, unchecked: 0 }; const clauses = items.filter(item => item.type === type); return { known: true, total: clauses.length, atRisk: clauses.filter(item => item.user_status === "at_risk").length, unchecked: clauses.filter(item => !item.user_status || item.user_status === "unchecked").length }; }
function WorkspaceContext({ overview, tab }) {
  const known = Array.isArray(overview.risk_items); const clauses = known ? overview.risk_items.filter(item => item.type !== "review") : []; const unchecked = clauses.filter(item => !item.user_status || item.user_status === "unchecked").length;
  return <><div className="timeline workspaceTimeline"><div className="eyebrow">주요 일정</div><div className="timelineItem"><span>제출 마감</span><strong>{overview.submission_deadline || "입찰공고 참조"}</strong></div><div className="timelineItem"><span>질의 마감</span><strong>{overview.inquiry_deadline || "RFP 본문 미기재"}</strong></div><div className="timelineItem"><span>실격·감점·불이익 조항</span><strong>{known ? `${clauses.length}건${unchecked > 0 ? ` · 미확인 ${unchecked}건` : ""}` : "분석 전"}</strong><small>{known ? `신뢰도 ${Math.round((overview.confidence || 0) * 100)}%` : "분석 결과를 불러오는 중입니다"}</small></div></div><div className="cardCriteria"><div className="eyebrow">{labels[tab]} 카드 생성 기준</div><p>{cardCriteria[tab]}</p></div></>;
}

function App() {
  const [docs, setDocs] = useState([]); const [selected, setSelected] = useState(null); const [tab, setTab] = useState("overview"); const [questionContextTab, setQuestionContextTab] = useState("overview"); const [llmProvider, setLlmProvider] = useState("gemini-lite");
  const [conversationId] = useState(() => { const saved = sessionStorage.getItem("rfp-conversation-id"); const value = saved || crypto.randomUUID(); sessionStorage.setItem("rfp-conversation-id", value); return value; });
  const [data, setData] = useState(null); const [analysisCache, setAnalysisCache] = useState({}); const [evidence, setEvidence] = useState(null); const [pageText, setPageText] = useState(""); const [pageCount, setPageCount] = useState(0); const [toc, setToc] = useState([]); const [docSearch, setDocSearch] = useState(""); const [searchResults, setSearchResults] = useState([]); const [question, setQuestion] = useState(""); const [chatHistory, setChatHistory] = useState([]); const [typingAnswer, setTypingAnswer] = useState(""); const [lastQuestion, setLastQuestion] = useState(""); const [listQuery, setListQuery] = useState(""); const [listStatus] = useState("all"); const [listDifficulty] = useState("all"); const [listSort, setListSort] = useState("date"); const [loading, setLoading] = useState(false); const [error, setError] = useState(""); const [docsLoading, setDocsLoading] = useState(true); const [docsError, setDocsError] = useState("");
  const loadDocuments = () => { setDocsLoading(true); setDocsError(""); api.documents().then(items => { setDocs(items); setSelected(current => current || items[0] || null); }).catch(() => setDocsError("RFP 목록을 불러오지 못했습니다. 백엔드 연결을 확인하세요.")).finally(() => setDocsLoading(false)); };
  useEffect(() => { loadDocuments(); }, []);
  useEffect(() => { const header = document.querySelector("header>div:first-child"); if (!header || header.querySelector(".evaluationLink")) return; const link = document.createElement("a"); link.href = "/internal/evaluation"; link.className = "evaluationLink"; link.textContent = "내부 평가"; header.appendChild(link); }, []);
  useEffect(() => { const nav = document.querySelector(".layout>nav"); const layout = nav?.parentElement; if (!nav || !layout || nav.querySelector(".fileListToggle")) return undefined; const toggle = document.createElement("button"); toggle.type = "button"; toggle.className = "fileListToggle"; toggle.innerHTML = "<span>☰</span>파일 목록"; toggle.setAttribute("aria-label", "파일 목록 열기"); toggle.setAttribute("aria-expanded", "false"); const sort = document.createElement("select"); sort.className = "fileListSort"; sort.setAttribute("aria-label", "파일 정렬"); sort.innerHTML = '<option value="date">날짜순</option><option value="title">이름순</option>'; sort.value = listSort; const applyOpen = open => { layout.classList.toggle("fileListCollapsed", !open); toggle.setAttribute("aria-expanded", String(open)); toggle.setAttribute("aria-label", open ? "파일 목록 접기" : "파일 목록 열기"); }; applyOpen(false); toggle.addEventListener("click", () => applyOpen(layout.classList.contains("fileListCollapsed"))); sort.addEventListener("change", event => setListSort(event.target.value)); nav.prepend(sort); nav.prepend(toggle); return () => { toggle.remove(); sort.remove(); layout.classList.remove("fileListCollapsed"); }; }, []);
  useEffect(() => { const card = document.querySelector(".uploadCard"); if (!card) return undefined; const input = document.createElement("input"); input.type = "file"; input.accept = "application/pdf,.pdf"; input.hidden = true; card.appendChild(input); const choose = () => input.click(); const upload = async () => { const file = input.files?.[0]; if (!file) return; setLoading(true); setError(""); try { await api.upload(file, file.name.replace(/\.pdf$/i, "")); const items = await api.documents(); setDocs(items); const uploaded = items.find(item => item.title === file.name.replace(/\.pdf$/i, "")); if (uploaded) setSelected(uploaded); } catch (e) { setError(e.message); } finally { setLoading(false); input.value = ""; } }; card.addEventListener("click", choose); input.addEventListener("change", upload); return () => { card.removeEventListener("click", choose); input.removeEventListener("change", upload); input.remove(); }; }, [docs.length]);
  useEffect(() => { if (!selected) return; api.toc(selected.document_id).then(result => setToc(result.items || [])).catch(() => setToc([])); }, [selected]);
  useEffect(() => { if (!selected || tab === "chat") return; const key = `${selected.document_id}:${tab}`; if (analysisCache[key]) { setData(analysisCache[key]); return; } const loaders = { overview: async id => { const [summary, riskData, deliverableData, eligibilityData, requirementData] = await Promise.all([api.overview(id), api.risks(id), api.deliverables(id), api.eligibility(id), api.requirements(id)]); const items = deliverableData.items || []; return { ...summary, risk_items: riskData.risks || [], deliverable_items: items, eligibility_items: eligibilityData.items || [], requirement_items: requirementData.items || [], deliverable_progress: { completed: items.filter(item => item.status === "completed").length, total: items.length } }; }, go: async id => ({ ...(await api.risks(id)), eligibility: (await api.eligibility(id)).items }), deliverables: api.deliverables, requirements: api.requirements }; setLoading(true); setError(""); loaders[tab](selected.document_id).then(result => { setData(result); setAnalysisCache(prev => ({ ...prev, [key]: result, ...(tab === "overview" && result.deliverable_items ? { [`${selected.document_id}:deliverables`]: { document_id: selected.document_id, items: result.deliverable_items }, [`${selected.document_id}:go`]: { document_id: selected.document_id, risks: result.risk_items, eligibility: result.eligibility_items }, [`${selected.document_id}:requirements`]: { document_id: selected.document_id, items: result.requirement_items } } : {}) })); }).catch(e => setError(e.message)).finally(() => setLoading(false)); }, [selected, tab, analysisCache]);
  useEffect(() => { if (tab !== "chat") setQuestionContextTab(tab); }, [tab]);
  useEffect(() => { if (tab === "chat") { setData(null); setTypingAnswer(""); setError(""); } }, [tab]);
  const askQuestion = async text => { const clean = text.trim(); if (!clean || !selected) return; setQuestion(clean); setLastQuestion(clean); setLoading(true); setError(""); setTypingAnswer(""); try { const result = await streamAsk(selected.document_id, clean, chatHistory.slice(-6), llmProvider, conversationId, delta => setTypingAnswer(delta)); setData(result); setTypingAnswer(""); setEvidence(result.citations?.[0] ? { document_name: result.citations[0].document_name, page_number: result.citations[0].page_start, quote: result.citations[0].quote || "인용된 검색 청크를 확인하세요.", score: result.citations[0].score ?? 1, requirement_ids: result.citations[0].requirement_ids } : null); setChatHistory(prev => [...prev, { role: "user", content: clean }, { role: "assistant", content: result.answer }].slice(-6)); } catch (e) { setError(e.message); } finally { setLoading(false); } };
  const ask = async event => { event.preventDefault(); await askQuestion(question); };
  const resetConversation = async () => { if (!selected) return; try { await api.resetConversation(selected.document_id, conversationId); setChatHistory([]); setData(null); setQuestion(""); setLastQuestion(""); setTypingAnswer(""); setEvidence(null); setError(""); } catch (e) { setError(e.message); } };
  const updateEligibility = async (item, status) => { try { await api.updateEligibility(selected.document_id, item.id, status); setData(prev => ({ ...prev, items: prev.items ? prev.items.map(row => row.id === item.id ? { ...row, user_status: status } : row) : prev.items, eligibility: (prev.eligibility || []).map(row => row.id === item.id ? { ...row, user_status: status } : row) })); } catch (e) { setError(e.message); } };
  const updateRisk = async (item, status) => { try { await api.updateRisk(selected.document_id, item.id, status); const update = rows => (rows || []).map(row => row.id === item.id ? { ...row, user_status: status } : row); setData(prev => ({ ...prev, risks: update(prev.risks), risk_items: update(prev.risk_items) })); setAnalysisCache(prev => { const next = { ...prev }; ["overview", "go"].forEach(suffix => { const key = `${selected.document_id}:${suffix}`; if (next[key]) next[key] = { ...next[key], risks: update(next[key].risks), risk_items: update(next[key].risk_items) }; }); return next; }); } catch (e) { setError(e.message); } };
  const updateDeliverable = async (item, status) => { try { await api.updateDeliverable(selected.document_id, item.id, { status }); setData(prev => ({ ...prev, items: prev.items.map(row => row.id === item.id ? { ...row, status } : row) })); } catch (e) { setError(e.message); } };
  const showPage = page => { if (!selected || page < 1) return; api.page(selected.document_id, page).then(result => { setPageText(result.text); setPageCount(result.page_count); setEvidence(prev => ({ ...(prev || { document_name: selected.title, score: 0, requirement_ids: [] }), page_number: page })); }).catch(e => setError(e.message)); };
  const showEvidence = item => { setEvidence(item); if (item?.page_number) showPage(item.page_number); };
  const runDocSearch = async event => { event.preventDefault(); if (!selected || !docSearch.trim()) return; try { const result = await api.searchPages(selected.document_id, docSearch); setSearchResults(result.results || []); } catch (e) { setError(e.message); } };
  const answerDocSearch = () => { if (!selected || !docSearch.trim()) return; const text = docSearch; setTab("chat"); setSearchResults([]); askQuestion(text); };
  const visibleDocs = docs.filter(doc => doc.title.toLowerCase().includes(listQuery.toLowerCase())).sort((a, b) => listSort === "title" ? a.title.localeCompare(b.title) : String(b.document_date || b.document_id).localeCompare(String(a.document_date || a.document_id)));
  const overview = analysisCache[`${selected?.document_id}:overview`] || (tab === "overview" ? data : null) || {}; const goData = analysisCache[`${selected?.document_id}:go`] || (tab === "go" ? data : null); const riskItems = goData?.risks ?? overview.risk_items; const disqualification = riskSummary(riskItems, "disqualification"); const deduction = riskSummary(riskItems, "deduction"); const riskCounts = { critical: disqualification.known ? disqualification.total : "분석 전", warning: deduction.known ? deduction.total : "분석 전" }; const deliverableData = analysisCache[`${selected?.document_id}:deliverables`] || (tab === "deliverables" ? data : null); const progress = deliverableData?.items ? { completed: deliverableData.items.filter(item => item.status === "completed").length, total: deliverableData.items.length } : (overview.deliverable_progress || {}); suggestions.chat = suggestions[questionContextTab] || suggestions.chat;
  activeDocumentId = selected?.document_id || "";
  return <div className="app"><header><div><strong>RFP Action Copilot</strong><span> 실제 RFP 검토 MVP</span></div><div className="statusbar"><span>제출 {overview.submission_deadline || "공고 참조"}</span><span>질의 {overview.inquiry_deadline || "본문 미기재"}</span><span className="danger">실격 {riskCounts.critical || 0}</span><span className="warning">감점·불이익 {riskCounts.warning || 0}</span><span>제출물 {progress.completed || 0}/{progress.total || 0}</span></div></header><div className="layout"><nav><h2>RFP 목록</h2><input className="listSearch" value={listQuery} onChange={e => setListQuery(e.target.value)} placeholder="RFP 검색"/><div className="listControls"><select value={listSort} onChange={e => setListSort(e.target.value)}><option value="difficulty">난이도순</option><option value="title">이름순</option></select><select value={listStatus} onChange={e => setListStatus(e.target.value)}><option value="all">전체 상태</option><option value="ready">준비 완료</option></select><select value={listDifficulty} onChange={e => setListDifficulty(e.target.value)}><option value="all">전체 난이도</option><option value="high">고난도</option><option value="medium">중간</option><option value="low">저난도</option></select></div>{docsLoading && <div className="empty">RFP 목록 불러오는 중…</div>}{docsError && <div className="error">{docsError}<button className="retry" onClick={loadDocuments}>재시도</button></div>}{!docsLoading && !docsError && visibleDocs.length === 0 && <div className="empty">표시할 RFP가 없습니다.</div>}{visibleDocs.map(doc => <button className={selected?.document_id === doc.document_id ? "doc active" : "doc"} onClick={() => { setSelected(doc); setTab("overview"); setData(null); setPageText(""); setSearchResults([]); setDocSearch(""); setChatHistory([]); }} key={doc.document_id}><b>{doc.title}</b><small>{doc.document_id} · {doc.organization}</small><em>마감: 공고 참조 · 상태: {doc.status || "ready"}</em></button>)}<div className="uploadCard">+ RFP 업로드<br/><small>PDF 파일 선택</small></div></nav><main><div className="workspaceHead"><div><div className="eyebrow">WORKSPACE</div><h1>{selected?.title || "문서를 선택하세요"}</h1></div><span className="badge">{selected?.difficulty || "-"}</span></div><div className="tabs">{tabs.map(item => <button className={tab === item ? "tab active" : "tab"} onClick={() => { setTab(item); if (item !== "chat") setData(analysisCache[`${selected?.document_id}:${item}`] || null); }} key={item}>{labels[item]}</button>)}</div><div className="mainBody"><Toc items={toc} onSelect={showPage}/><section className="content"><WorkspaceContext overview={overview} tab={tab}/><form className="docSearch" onSubmit={runDocSearch}><input value={docSearch} onChange={e => setDocSearch(e.target.value)} placeholder="키워드로 원문 페이지 검색"/><button type="submit">원문 찾기</button><button type="button" className="answerSearch" onClick={answerDocSearch}>AI로 답변</button></form>{searchResults.length > 0 && <div className="searchResults">{searchResults.map(result => <button key={result.page} onClick={() => { showPage(result.page); setSearchResults([]); setDocSearch(""); }}><b>p.{result.page}</b> {result.excerpt}</button>)}</div>}{error && <div className="error">{error}{tab === "chat" && <button className="retry" onClick={() => askQuestion(lastQuestion)}>재시도</button>}</div>}{loading && <div className="loading">분석 중… ({llmProvider === "openai" ? "ChatGPT nano" : llmProvider === "gemini" ? "Gemini 3.5 Flash" : "Gemini 3.5 Flash-Lite"} 호출)</div>}{!loading && tab === "chat" && <><form className="chat" onSubmit={ask}><div className="chatInput"><select aria-label="LLM 선택" value={llmProvider} onChange={e => setLlmProvider(e.target.value)}><option value="openai">ChatGPT nano</option><option value="gemini">Gemini 3.5 Flash</option><option value="gemini-lite">Gemini 3.5 Flash-Lite</option></select><textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="예: 실격 조건과 필수 제출물을 알려줘" /></div><button>질문하기</button><button type="button" className="resetChat" onClick={resetConversation}>대화 초기화</button></form><div className="suggestions">{suggestions.chat.map(item => <button type="button" onClick={() => askQuestion(item)} key={item}>{item}</button>)}</div></>}{!loading && data && tab !== "chat" && <DataView data={data} tab={tab} onEvidence={showEvidence} onEligibility={updateEligibility} onDeliverable={updateDeliverable} />}{!loading && data && tab === "chat" && <><DataView data={{ ...data, answer: typingAnswer || data.answer }} onEvidence={showEvidence} /><div className="suggestions">{suggestions.chat.map(item => <button type="button" onClick={() => askQuestion(item)} key={item}>{item}</button>)}</div></>}</section></div></main><Evidence evidence={evidence} pageText={pageText} pageCount={pageCount} onPage={showPage} /></div></div>;
}

function DataView({ data, tab, onEvidence, onEligibility, onRisk, onDeliverable }) {
  if (tab === "overview") return <OverviewView data={data} onEvidence={onEvidence} />;
  if (tab === "go") return <GoNoGoView data={data} onEvidence={onEvidence} onEligibility={onEligibility} onRisk={onRisk} />;
  if (tab === "deliverables") return <DeliverablesView data={data} onEvidence={onEvidence} onDeliverable={onDeliverable} />;
  if (tab === "requirements") return <RequirementsView data={data} onEvidence={onEvidence} />;
  if (data.answer) return <div className="answer"><div className="eyebrow">ANSWER</div><p>{data.answer}</p>{data.confidence !== null && data.confidence !== undefined && <small>근거 신뢰도 {Math.round(data.confidence * 100)}%</small>}{data.caveat && <p className="caveat">{data.caveat}</p>}<div className="chips">{data.citations?.map((c, index) => <button onClick={() => onEvidence({ document_name: c.document_name, page_number: c.page_start, quote: c.quote || "인용된 검색 청크를 확인하세요.", score: c.score ?? 1, requirement_ids: c.requirement_ids })} key={`${c.chunk_id}-${index}`}>{c.document_id} · p.{c.page_start}</button>)}</div></div>;
  const items = data.items || data.risks || data.action_items || [];
  return <div className="cards">{items.length === 0 ? <div className="empty">분석 결과가 없습니다.</div> : items.map((item, index) => { const score = item.evidence?.score; const riskClass = tab === "risks" ? `risk-${item.severity || "info"}` : ""; return <div className={`card ${riskClass}`} key={item.id || index}><button className="cardBody" onClick={() => onEvidence(item.evidence)}><strong>{item.title || item.name || item.description}</strong><p>{item.description || item.type || item.status || ""}</p>{tab === "deliverables" && <small>{item.format || "형식 확인 필요"} · 수량 {item.quantity || 1} · 담당 {item.assignee || "미배정"}{item.deadline ? ` · ${item.deadline}` : ""}</small>}{score !== undefined && score < 0.7 && <span className="lowConfidence">⚠️ 낮은 신뢰도</span>}</button><div className="cardFooter"><EvidenceButton evidence={item.evidence} onEvidence={onEvidence}/>{tab === "eligibility" && <select value={item.user_status || "unchecked"} onChange={e => onEligibility(item, e.target.value)}><option value="unchecked">미확인</option><option value="met">충족</option><option value="not_met">미충족</option><option value="review_required">확인 필요</option></select>}{tab === "deliverables" && <select value={item.status || "pending"} onChange={e => onDeliverable(item, e.target.value)}><option value="pending">미착수</option><option value="in_progress">진행 중</option><option value="completed">완료</option></select>}</div></div>; })}</div>;
}

function DeliverablesView({ data, onEvidence, onDeliverable }) {
  const items = data.items || []; const completed = items.filter(item => item.status === "completed").length; const inProgress = items.filter(item => item.status === "in_progress").length;
  const groups = [{ kind: "bid_submission", title: "입찰 제출서류", description: "입찰·제안 단계에서 제출할 서류" }, { kind: "project_deliverable", title: "사업 수행 산출물", description: "계약 이후 작성·납품할 계획서, 보고서와 성과품" }];
  return <div className="deliverablesView"><div className="deliverableSummary"><div><small>전체 항목</small><strong>{items.length}개</strong></div><div><small>완료</small><strong>{completed}개</strong></div><div><small>진행 중</small><strong>{inProgress}개</strong></div><div><small>미완료</small><strong>{items.length - completed}개</strong></div></div>{items.length === 0 ? <div className="empty">문서에서 확인된 제출물·산출물이 없습니다.</div> : groups.map(group => { const rows = items.filter(item => (item.kind || "bid_submission") === group.kind); return <section className="deliverableGroup" key={group.kind}><div className="sectionTitle">{group.title} <span>{rows.length}개</span></div><p className="sectionDescription">{group.description}</p><div className="cards">{rows.length === 0 ? <div className="empty">해당 항목 없음</div> : rows.map((item, index) => <div className="card" key={item.id || index}><button className="cardBody" onClick={() => onEvidence(item.evidence)}><strong>{item.name || item.title}</strong><p>{item.description}</p><small>{item.format || "형식 확인 필요"} · 수량 {item.quantity || 1} · 담당 {item.assignee || "미배정"}{item.deadline ? ` · ${item.deadline}` : ""}</small></button><div className="cardFooter"><EvidenceButton evidence={item.evidence} onEvidence={onEvidence}/><select value={item.status || "pending"} onChange={e => onDeliverable(item, e.target.value)}><option value="pending">미착수</option><option value="in_progress">진행 중</option><option value="completed">완료</option></select></div></div>)}</div></section>; })}</div>;
}

const requirementGroups = [
  { id: "all", label: "전체", description: "문서에서 확인된 모든 요구사항" },
  { id: "product", label: "기능", categories: ["functional"], description: "구현해야 할 업무 기능과 사용자 흐름" },
  { id: "quality", label: "품질 · 보안", categories: ["performance", "security", "quality", "interface", "data"], description: "성능, 보안, 데이터와 연계 품질 기준" },
  { id: "delivery", label: "수행 · 계약", categories: ["operation", "personnel", "output", "deliverable", "contract", "project"], description: "운영, 인력, 산출물과 계약상 의무" },
];
const requirementCategoryLabels = { functional: "기능", performance: "성능", security: "보안", quality: "품질", interface: "연계", data: "데이터", operation: "운영", personnel: "인력", output: "산출물", deliverable: "산출물", contract: "계약", project: "사업관리" };
const requirementStatusLabels = { reviewed: "검토 완료", pending: "미검토", flagged: "확인 필요" };
const requirementPriorityLabels = { high: "필수", medium: "중요", low: "참고" };

function RequirementsView({ data, onEvidence }) {
  const [group, setGroup] = useState("all");
  const [query, setQuery] = useState("");
  const items = data.items || [];
  const activeGroup = requirementGroups.find(item => item.id === group) || requirementGroups[0];
  const groupedCount = candidate => candidate.id === "all" ? items.length : items.filter(item => candidate.categories.includes(item.category)).length;
  const visible = items.filter(item => group === "all" || activeGroup.categories.includes(item.category)).filter(item => `${item.id || ""} ${item.title || ""} ${item.description || ""}`.toLowerCase().includes(query.toLowerCase()));
  const sections = (group === "all" ? requirementGroups.slice(1) : [activeGroup]).map(candidate => ({ ...candidate, items: visible.filter(item => candidate.categories.includes(item.category)) })).filter(section => section.items.length > 0);
  const uncategorized = group === "all" ? visible.filter(item => !requirementGroups.slice(1).some(candidate => candidate.categories.includes(item.category))) : [];

  return <div className="requirementsView">
    <div className="requirementsIntro"><div><div className="eyebrow">REQUIREMENT MAP</div><h2>이 문서가 요구하는 것</h2><p>비슷한 성격의 요구사항을 묶었습니다. 항목을 선택하면 바로 원문 근거를 확인할 수 있습니다.</p></div><div className="requirementSummary"><strong>{items.length}</strong><span>전체 요구사항</span><small>필수 {items.filter(item => item.priority === "high").length} · 확인 필요 {items.filter(item => item.review_status === "flagged").length}</small></div></div>
    <div className="requirementToolbar"><div className="requirementGroupTabs" role="tablist" aria-label="요구사항 성격"><span className="flowLabel">성격별 보기</span>{requirementGroups.map(item => <button role="tab" aria-selected={group === item.id} className={group === item.id ? "active" : ""} onClick={() => setGroup(item.id)} key={item.id}>{item.label}<b>{groupedCount(item)}</b></button>)}</div><input value={query} onChange={event => setQuery(event.target.value)} placeholder="요구사항 ID 또는 내용 검색" aria-label="요구사항 검색"/></div>
    {visible.length === 0 && <div className="empty">조건에 맞는 요구사항이 없습니다.</div>}
    {[...sections, ...(uncategorized.length ? [{ id: "other", label: "기타", description: "별도 분류가 필요한 요구사항", items: uncategorized }] : [])].map(section => <section className="requirementSection" key={section.id}><div className="requirementSectionHead"><div><h3>{section.label}</h3><p>{section.description}</p></div><span>{section.items.length}개</span></div><div className="requirementList">{section.items.map((item, index) => <article className="requirementItem" key={item.id || index}><button className="requirementMain" onClick={() => onEvidence(item.evidence)}><span className="requirementId">{item.id || `REQ-${index + 1}`}</span><div><strong>{item.title || item.description}</strong><p>{item.description}</p><div className="requirementMeta"><span>{requirementCategoryLabels[item.category] || item.category || "기타"}</span><span className={`priority-${item.priority || "medium"}`}>{requirementPriorityLabels[item.priority] || item.priority || "중요"}</span><span>{requirementStatusLabels[item.review_status] || item.review_status || "미검토"}</span></div></div></button><EvidenceButton evidence={item.evidence} onEvidence={onEvidence}/></article>)}</div></section>)}
  </div>;
}

function GoNoGoView({ data, onEvidence, onEligibility, onRisk }) {
  const eligibilityItems = data.eligibility || [];
  const [riskItems, setRiskItems] = useState(data.risks || []);
  const setRisk = async (item, status) => {
    await api.updateRisk(activeDocumentId, item.id, status);
    setRiskItems(rows => rows.map(row => row.id === item.id ? { ...row, user_status: status } : row));
    if (onRisk) onRisk(item, status);
  };
  const dq = riskSummary(riskItems, "disqualification");
  const dd = riskSummary(riskItems, "deduction");
  const totalRiskClauses = dq.total + dd.total;
  const eligibility = {
    total: eligibilityItems.length,
    notMet: eligibilityItems.filter(item => item.user_status === "not_met").length,
    unchecked: eligibilityItems.filter(item => !item.user_status || item.user_status === "unchecked" || item.user_status === "review_required").length,
  };
  const eligibilityEvidence = eligibilityItems.find(item => item.evidence)?.evidence;
  const disqualificationEvidence = riskItems.find(item => item.type === "disqualification" && item.evidence)?.evidence;
  const deductionEvidence = riskItems.find(item => item.type === "deduction" && item.evidence)?.evidence;
  const evidenceCardProps = evidence => evidence ? {
    role: "button", tabIndex: 0,
    onClick: () => onEvidence(evidence),
    onKeyDown: event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); onEvidence(evidence); } },
  } : {};

  return <div className="goNoGoView">
    <div className="riskRatioSummary">
      <div className="eligibilitySummary" {...evidenceCardProps(eligibilityEvidence)}><small>참가 자격 미충족 / 전체 조항</small><strong>{eligibility.notMet}/{eligibility.total}</strong><span>미확인·확인 필요 {eligibility.unchecked}</span></div>
      <div className="disqualificationSummary" {...evidenceCardProps(disqualificationEvidence)}><small>실격 조항 / 전체 위험 조항</small><strong>{dq.total}/{totalRiskClauses}</strong><span>위험 있음 {dq.atRisk} · 미확인 {dq.unchecked}</span></div>
      <div className="deductionSummary" {...evidenceCardProps(deductionEvidence)}><small>감점·불이익 / 전체 위험 조항</small><strong>{dd.total}/{totalRiskClauses}</strong><span>위험 있음 {dd.atRisk} · 미확인 {dd.unchecked}</span></div>
    </div>
    <div className="sectionTitle">참가 자격</div>
    <div className="cards">{eligibilityItems.map(item => <div className="card" key={item.id}><button className="cardBody" onClick={() => onEvidence(item.evidence)}><strong>{item.title}</strong><p>{item.description}</p></button><div className="cardFooter"><EvidenceButton evidence={item.evidence} onEvidence={onEvidence}/><select value={item.user_status || "unchecked"} onChange={e => onEligibility(item, e.target.value)}><option value="unchecked">미확인</option><option value="met">충족</option><option value="not_met">미충족</option><option value="review_required">확인 필요</option></select></div></div>)}</div>
    <div className="sectionTitle">실격·감점·불이익 조항</div>
    <div className="cards">{riskItems.map(item => <div className={`card risk-${item.severity || "info"}`} key={item.id}><button className="cardBody" onClick={() => onEvidence(item.evidence)}><strong>{item.title}</strong><p>{item.description}</p></button><div className="cardFooter"><EvidenceButton evidence={item.evidence} onEvidence={onEvidence}/><select value={item.user_status || "unchecked"} onChange={e => setRisk(item, e.target.value)}><option value="unchecked">미확인</option><option value="at_risk">위험 있음</option><option value="safe">문제없음</option><option value="review_required">추가 확인</option></select></div></div>)}</div>
  </div>;
}

function OverviewView({ data, onEvidence }) {
  const dq = riskSummary(data.risk_items, "disqualification"); const dd = riskSummary(data.risk_items, "deduction"); const progress = data.deliverable_progress || {};
  const eligibility = data.eligibility_items || []; const requirements = data.requirement_items || []; const deliverables = data.deliverable_items || [];
  const uncheckedEligibility = eligibility.filter(item => !item.user_status || ["unchecked", "review_required"].includes(item.user_status)).length;
  const requirementCounts = requirements.reduce((counts, item) => ({ ...counts, [item.category]: (counts[item.category] || 0) + 1 }), {});
  useEffect(() => { const cards = [...document.querySelectorAll(".tabSummaryGrid article")]; const handlers = cards.map((card, index) => { const activate = event => { if (event.type === "keydown" && !["Enter", " "].includes(event.key)) return; event.preventDefault(); document.querySelectorAll(".tabs .tab")[index + 1]?.click(); }; card.tabIndex = 0; card.setAttribute("role", "button"); card.addEventListener("click", activate); card.addEventListener("keydown", activate); return activate; }); return () => cards.forEach((card, index) => { card.removeEventListener("click", handlers[index]); card.removeEventListener("keydown", handlers[index]); }); }, []);
  useEffect(() => {
    const actionEvidence = (data.action_items || []).find(item => item.evidence)?.evidence;
    const sources = [
      actionEvidence,
      actionEvidence,
      eligibility.find(item => item.evidence)?.evidence,
      (data.risk_items || []).find(item => item.type === "disqualification" && item.evidence)?.evidence,
      (data.risk_items || []).find(item => item.type === "deduction" && item.evidence)?.evidence,
      deliverables.find(item => item.evidence)?.evidence,
    ];
    const cards = [...document.querySelectorAll(".overview .summaryCard")];
    const bindings = cards.map((card, index) => {
      const source = sources[index];
      if (!source) return null;
      const activate = event => { if (event.type === "keydown" && !["Enter", " "].includes(event.key)) return; event.preventDefault(); onEvidence(source); };
      card.tabIndex = 0; card.setAttribute("role", "button");
      card.addEventListener("click", activate); card.addEventListener("keydown", activate);
      return activate;
    });
    return () => cards.forEach((card, index) => { const handler = bindings[index]; if (!handler) return; card.removeEventListener("click", handler); card.removeEventListener("keydown", handler); card.removeAttribute("role"); card.removeAttribute("tabindex"); });
  }, [data, eligibility, deliverables, onEvidence]);
  return <div className="overview"><div className="summaryGrid"><div className="summaryCard"><small>제출 마감</small><strong>{data.submission_deadline || "입찰공고 참조"}</strong></div><div className="summaryCard"><small>질의 마감</small><strong>{data.inquiry_deadline || "RFP 본문 미기재"}</strong></div><div className="summaryCard"><small>자격 충족</small><strong>{data.eligibility_summary || "확인 필요"}</strong></div><div className="summaryCard severity-critical"><small>실격 조항</small><strong>{dq.total}건</strong><span>위험 있음 {dq.atRisk} · 미확인 {dq.unchecked}</span></div><div className="summaryCard severity-warning"><small>감점·불이익</small><strong>{dd.total}건</strong><span>위험 있음 {dd.atRisk} · 미확인 {dd.unchecked}</span></div><div className="summaryCard"><small>제출물 완료율</small><strong>{progress.completed || 0}/{progress.total || 0}</strong><progress value={progress.completed || 0} max={progress.total || 1}/></div></div><div className="sectionTitle">탭별 핵심 요약</div><div className="tabSummaryGrid"><article><small>GO / NO-GO</small><strong>자격 {eligibility.length}건 · 위험 {dq.total + dd.total}건</strong><p>미확인 자격 {uncheckedEligibility}건, 실격 {dq.total}건, 감점·불이익 {dd.total}건을 우선 확인하세요.</p></article><article><small>실행 준비</small><strong>제출물 {deliverables.length}건</strong><p>완료 {progress.completed || 0}건 · 미완료 {Math.max(0, deliverables.length - (progress.completed || 0))}건입니다. 원본·날인·수량 조건을 함께 확인하세요.</p></article><article><small>요구사항</small><strong>전체 {requirements.length}건</strong><p>기능 {requirementCounts.functional || 0} · 성능 {requirementCounts.performance || 0} · 보안 {requirementCounts.security || 0} · 수행/계약 {(requirementCounts.operation || 0) + (requirementCounts.personnel || 0) + (requirementCounts.output || 0) + (requirementCounts.contract || 0)}</p></article><article><small>AI 질문</small><strong>문서 근거 기반 확인</strong><p>마감, 참가 자격, 실격·불이익, 제출물과 요구사항을 질문하고 인용 페이지에서 원문을 검증할 수 있습니다.</p></article></div><div className="sectionTitle">우선 조치 항목</div><div className="cards">{(data.action_items || []).length === 0 ? <div className="empty">요약 정보는 확인되었지만 즉시 조치 항목은 추출되지 않았습니다.</div> : data.action_items.map((item, index) => <div className="card" key={item.id || index}><button className="cardBody" onClick={() => onEvidence(item.evidence)}><strong>{item.title}</strong><p>{item.description}</p><small>{item.due_date || "마감일 확인 필요"} · {item.priority || "info"}</small></button><EvidenceButton evidence={item.evidence} onEvidence={onEvidence}/></div>)}</div></div>;
}

function EvaluationPage() {
  const [summary, setSummary] = useState(null); const [error, setError] = useState("");
  useEffect(() => { api.evaluationSummary().then(setSummary).catch(e => setError(e.message)); }, []);
  const rows = summary?.retrieval || [];
  return <div className="evaluationPage"><header className="evaluationHeader"><div><strong>RFP Action Copilot</strong><span> INTERNAL EVALUATION</span></div><a href="/">사용자 화면으로 돌아가기</a></header><main><div className="eyebrow">EVALUATION CONTROL ROOM</div><h1>검색·답변 품질 평가</h1><p className="evaluationIntro">Golden set 기반 검색 평가와 RAGAS 실행 결과를 확인하는 내부 운영팀 전용 화면입니다.</p>{error && <div className="error">{error}</div>}<div className="evaluationGrid"><div className="evaluationCard"><small>평가 상태</small><strong>{summary?.status === "ready" ? "결과 있음" : "미실행"}</strong><p>검색 평가 summary.json</p></div><div className="evaluationCard"><small>평가 대상</small><strong>{rows.length || 0}개 retriever</strong><p>BM25 · Dense · Hybrid · Reranker</p></div><div className="evaluationCard"><small>RAGAS</small><strong>Golden set 대기</strong><p>입력 변환 스크립트 준비됨</p></div></div><section className="evaluationSection"><h2>검색 평가 결과</h2>{rows.length === 0 ? <div className="empty">아직 평가 결과가 없습니다. Golden set을 준비한 뒤 evaluate_retrieval을 실행하세요.</div> : <table><thead><tr><th>Retriever</th><th>Queries</th><th>Recall@k</th><th>MRR</th></tr></thead><tbody>{rows.map((row, index) => <tr key={row.retriever || index}><td>{row.retriever || row.name || `result-${index + 1}`}</td><td>{row.query_count ?? "-"}</td><td>{row.recall_at_k ?? row.recall ?? "-"}</td><td>{row.mrr ?? "-"}</td></tr>)}</tbody></table>}</section><section className="evaluationSection"><h2>실행 명령</h2><pre>uv run python -m scripts.evaluate_retrieval --golden data/eval/golden_set.jsonl --retriever all
uv run python -m scripts.prepare_ragas_dataset --input data/eval/golden_set.jsonl</pre></section></main></div>;
}

function Root() { return window.location.pathname === "/internal/evaluation" ? <PipelineEvaluationPage /> : <App />; }
createRoot(document.getElementById("root")).render(<Root />);
