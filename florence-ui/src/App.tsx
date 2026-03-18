import React, { useMemo, useState } from "react";

type HealthResponse = {
  status: string;
  indexed_chunks?: number;
  cached_documents?: number;
};

type QueryResultItem = {
  document: string;
  metadata?: {
    source?: string;
    page?: number | null;
    chunk_id?: number | null;
    [key: string]: unknown;
  };
  distance?: number | null;
};

type QueryResponse = {
  query: string;
  results: QueryResultItem[];
};

type CitationItem = {
  source?: string | null;
  chunk_id?: number | null;
  page?: number | null;
  score?: number | null;
  excerpt: string;
};

type AskResponse = {
  query: string;
  answer: string;
  extracted_answer: string;
  answer_score: number;
  citations: CitationItem[];
};

type IngestResponse = {
  source?: string;
  chunks_ingested?: number;
  [key: string]: unknown;
};

type ApiResult = QueryResponse | AskResponse | IngestResponse | Record<string, unknown> | null;

export default function App() {
  const defaultApiBase = "http://127.0.0.1:8000";

  const [tab, setTab] = useState<"ingest" | "query" | "ask">("ask");
  const [apiBase, setApiBase] = useState(defaultApiBase);

  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(3);

  const [sourceName, setSourceName] = useState("manual_note.txt");
  const [text, setText] = useState("");
  const [ingestFile, setIngestFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [healthLoading, setHealthLoading] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [result, setResult] = useState<ApiResult>(null);
  const [error, setError] = useState("");
  const [lastAction, setLastAction] = useState("");

  const endpointHint = useMemo(() => {
    if (tab === "ingest") return "/ingest-file or /ingest-text";
    if (tab === "query") return "/query";
    return "/ask";
  }, [tab]);

  const normalizedApiBase = apiBase.replace(/\/+$/, "");

  const askResult = result as AskResponse | null;
  const queryResult = result as QueryResponse | null;
  const ingestResult = result as IngestResponse | null;

  async function parseJsonResponse(res: Response) {
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      return res.json();
    }
    return { detail: await res.text() };
  }

  async function callHealth() {
    setHealthLoading(true);
    setError("");
    try {
      const res = await fetch(`${normalizedApiBase}/health`);
      const data = (await parseJsonResponse(res)) as HealthResponse | { detail?: string };

      if (!res.ok) {
        throw new Error((data as { detail?: string }).detail || `HTTP ${res.status}`);
      }

      setHealth(data as HealthResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setHealthLoading(false);
    }
  }

  async function handleIngestText() {
    setLoading(true);
    setError("");
    setResult(null);
    setLastAction("ingest-text");

    try {
      const form = new FormData();
      form.append("text", text);
      form.append("source", sourceName || "manual_note.txt");

      const res = await fetch(`${normalizedApiBase}/ingest-text`, {
        method: "POST",
        body: form,
      });

      const data = await parseJsonResponse(res);
      if (!res.ok) {
        throw new Error((data as { detail?: string }).detail || `HTTP ${res.status}`);
      }

      setResult(data as IngestResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleIngestFile() {
    setLoading(true);
    setError("");
    setResult(null);
    setLastAction("ingest-file");

    try {
      if (!ingestFile) throw new Error("Selecione um arquivo para ingestão.");

      const form = new FormData();
      form.append("file", ingestFile);

      const res = await fetch(`${normalizedApiBase}/ingest-file`, {
        method: "POST",
        body: form,
      });

      const data = await parseJsonResponse(res);
      if (!res.ok) {
        throw new Error((data as { detail?: string }).detail || `HTTP ${res.status}`);
      }

      setResult(data as IngestResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleQuery() {
    setLoading(true);
    setError("");
    setResult(null);
    setLastAction("query");

    try {
      const form = new FormData();
      form.append("query", query);
      form.append("top_k", String(topK));

      const res = await fetch(`${normalizedApiBase}/query`, {
        method: "POST",
        body: form,
      });

      const data = await parseJsonResponse(res);
      if (!res.ok) {
        throw new Error((data as { detail?: string }).detail || `HTTP ${res.status}`);
      }

      setResult(data as QueryResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk() {
    setLoading(true);
    setError("");
    setResult(null);
    setLastAction("ask");

    try {
      const form = new FormData();
      form.append("query", query);
      form.append("top_k", String(topK));

      const res = await fetch(`${normalizedApiBase}/ask`, {
        method: "POST",
        body: form,
      });

      const data = await parseJsonResponse(res);
      if (!res.ok) {
        throw new Error((data as { detail?: string }).detail || `HTTP ${res.status}`);
      }

      setResult(data as AskResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function copyJson() {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
  }

  return (
      <div className="min-h-screen bg-slate-50 text-slate-900 p-6">
        <div className="max-w-7xl mx-auto grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-6">
            <div className="rounded-3xl bg-white shadow-sm border border-slate-200 p-6">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <h1 className="text-3xl font-semibold tracking-tight">Florence</h1>
                  <p className="text-slate-600 mt-2">
                    Interface para ingestão, consulta e resposta sobre a base RAG de documentos de saúde.
                  </p>
                </div>
                <button
                    onClick={callHealth}
                    className="rounded-2xl px-4 py-2 bg-slate-900 text-white hover:opacity-90 transition"
                >
                  {healthLoading ? "Verificando..." : "Checar API"}
                </button>
              </div>

              <div className="mt-5 grid md:grid-cols-[1fr_auto] gap-3 items-end">
                <label className="block">
                  <span className="text-sm font-medium text-slate-700">Base URL da API</span>
                  <input
                      value={apiBase}
                      onChange={(e) => setApiBase(e.target.value)}
                      className="mt-1 w-full rounded-2xl border border-slate-300 px-4 py-3 outline-none focus:ring-2 focus:ring-slate-300"
                      placeholder="http://127.0.0.1:8000"
                  />
                </label>
                <div className="text-sm text-slate-500">
                  Endpoint atual: <span className="font-mono">{endpointHint}</span>
                </div>
              </div>

              {health && (
                  <div className="mt-4 rounded-2xl bg-emerald-50 border border-emerald-200 p-4 text-sm">
                    <div>
                      <span className="font-medium">Status:</span> {health.status}
                    </div>
                    {health.indexed_chunks !== undefined && (
                        <div>
                          <span className="font-medium">Chunks indexados:</span> {health.indexed_chunks}
                        </div>
                    )}
                    {health.cached_documents !== undefined && (
                        <div>
                          <span className="font-medium">Documentos em cache:</span> {health.cached_documents}
                        </div>
                    )}
                  </div>
              )}
            </div>

            <div className="rounded-3xl bg-white shadow-sm border border-slate-200 p-3">
              <div className="grid grid-cols-3 gap-2">
                {[
                  ["ingest", "Ingestão"],
                  ["query", "Busca"],
                  ["ask", "Resposta"],
                ].map(([key, label]) => (
                    <button
                        key={key}
                        onClick={() => setTab(key as "ingest" | "query" | "ask")}
                        className={`rounded-2xl px-4 py-3 text-sm font-medium transition ${
                            tab === key ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                        }`}
                    >
                      {label}
                    </button>
                ))}
              </div>
            </div>

            {tab === "ingest" && (
                <div className="rounded-3xl bg-white shadow-sm border border-slate-200 p-6 space-y-6">
                  <div>
                    <h2 className="text-xl font-semibold">Ingestão</h2>
                    <p className="text-slate-600 mt-1">
                      Adicione texto manualmente ou envie um arquivo para indexação persistente.
                    </p>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="rounded-2xl border border-slate-200 p-4 space-y-4">
                      <div className="font-medium">Ingerir texto</div>
                      <label className="block">
                        <span className="text-sm text-slate-700">Nome da fonte</span>
                        <input
                            value={sourceName}
                            onChange={(e) => setSourceName(e.target.value)}
                            className="mt-1 w-full rounded-2xl border border-slate-300 px-4 py-3"
                            placeholder="protocolo_sepse.txt"
                        />
                      </label>
                      <label className="block">
                        <span className="text-sm text-slate-700">Texto</span>
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            rows={10}
                            className="mt-1 w-full rounded-2xl border border-slate-300 px-4 py-3"
                            placeholder="Cole aqui o texto do documento..."
                        />
                      </label>
                      <button
                          onClick={handleIngestText}
                          className="rounded-2xl px-4 py-3 bg-slate-900 text-white hover:opacity-90"
                          disabled={loading}
                      >
                        {loading && lastAction === "ingest-text" ? "Ingerindo..." : "Ingerir texto"}
                      </button>
                    </div>

                    <div className="rounded-2xl border border-slate-200 p-4 space-y-4">
                      <div className="font-medium">Ingerir arquivo</div>
                      <label className="block">
                        <span className="text-sm text-slate-700">Arquivo (.txt, .pdf, .docx)</span>
                        <input
                            type="file"
                            onChange={(e) => setIngestFile(e.target.files?.[0] || null)}
                            className="mt-1 w-full rounded-2xl border border-slate-300 px-4 py-3 bg-white"
                        />
                      </label>
                      <div className="rounded-2xl bg-slate-50 p-4 text-sm text-slate-600 min-h-24">
                        {ingestFile ? (
                            <div>
                              <div>
                                <span className="font-medium">Nome:</span> {ingestFile.name}
                              </div>
                              <div>
                                <span className="font-medium">Tamanho:</span> {Math.round(ingestFile.size / 1024)} KB
                              </div>
                              <div>
                                <span className="font-medium">Tipo:</span> {ingestFile.type || "desconhecido"}
                              </div>
                            </div>
                        ) : (
                            "Nenhum arquivo selecionado."
                        )}
                      </div>
                      <button
                          onClick={handleIngestFile}
                          className="rounded-2xl px-4 py-3 bg-slate-900 text-white hover:opacity-90"
                          disabled={loading}
                      >
                        {loading && lastAction === "ingest-file" ? "Enviando..." : "Ingerir arquivo"}
                      </button>
                    </div>
                  </div>
                </div>
            )}

            {tab !== "ingest" && (
                <div className="rounded-3xl bg-white shadow-sm border border-slate-200 p-6 space-y-5">
                  <div>
                    <h2 className="text-xl font-semibold">{tab === "query" ? "Busca semântica" : "Perguntar à base"}</h2>
                    <p className="text-slate-600 mt-1">
                      {tab === "query"
                          ? "Consulta a base persistida e retorna os chunks mais relevantes."
                          : "Consulta a base persistida, extrai a resposta e a reescreve com citações."}
                    </p>
                  </div>

                  <div className="grid md:grid-cols-[1fr_120px] gap-4">
                    <label className="block">
                      <span className="text-sm text-slate-700">Pergunta</span>
                      <input
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                          className="mt-1 w-full rounded-2xl border border-slate-300 px-4 py-3"
                          placeholder="Quais são os sinais de sepse?"
                      />
                    </label>
                    <label className="block">
                      <span className="text-sm text-slate-700">Top K</span>
                      <input
                          type="number"
                          value={topK}
                          min={1}
                          max={10}
                          onChange={(e) => setTopK(Number(e.target.value || 3))}
                          className="mt-1 w-full rounded-2xl border border-slate-300 px-4 py-3"
                      />
                    </label>
                  </div>

                  <button
                      onClick={tab === "query" ? handleQuery : handleAsk}
                      className="rounded-2xl px-4 py-3 bg-slate-900 text-white hover:opacity-90"
                      disabled={loading}
                  >
                    {loading ? "Executando..." : tab === "query" ? "Buscar" : "Responder"}
                  </button>
                </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="rounded-3xl bg-white shadow-sm border border-slate-200 p-6 min-h-[480px]">
              <div className="flex items-center justify-between gap-4 mb-4">
                <h2 className="text-xl font-semibold">Resultado</h2>
                {result && (
                    <button onClick={copyJson} className="rounded-2xl px-3 py-2 bg-slate-100 hover:bg-slate-200 text-sm">
                      Copiar JSON
                    </button>
                )}
              </div>

              {error && (
                  <div className="rounded-2xl bg-red-50 border border-red-200 p-4 text-red-700 text-sm mb-4">{error}</div>
              )}

              {!result && !error && (
                  <div className="rounded-2xl border border-dashed border-slate-300 p-8 text-center text-slate-500">
                    Nenhum resultado ainda.
                  </div>
              )}

              {askResult && "answer" in askResult && (
                  <div className="space-y-5">
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Resposta final</div>
                      <div className="text-lg leading-7">{askResult.answer}</div>
                    </div>

                    {askResult.extracted_answer && (
                        <div className="rounded-2xl border border-slate-200 p-4">
                          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Resposta extraída</div>
                          <div>{askResult.extracted_answer}</div>
                          <div className="mt-3 text-sm text-slate-500">
                            Score: {Number(askResult.answer_score).toFixed(4)}
                          </div>
                        </div>
                    )}

                    {Array.isArray(askResult.citations) && askResult.citations.length > 0 && (
                        <div className="space-y-3">
                          <div className="text-sm font-medium text-slate-700">Citações</div>
                          {askResult.citations.map((citation, idx) => (
                              <div key={idx} className="rounded-2xl border border-slate-200 p-4">
                                <div className="text-sm text-slate-600 mb-2">
                                  <span className="font-medium">Fonte:</span> {citation.source || "—"}
                                  {citation.page ? ` · Página ${citation.page}` : ""}
                                  {citation.chunk_id !== undefined && citation.chunk_id !== null
                                      ? ` · Chunk ${citation.chunk_id}`
                                      : ""}
                                  {citation.score !== undefined && citation.score !== null
                                      ? ` · Score ${Number(citation.score).toFixed(4)}`
                                      : ""}
                                </div>
                                <div className="text-sm leading-6 text-slate-800">{citation.excerpt}</div>
                              </div>
                          ))}
                        </div>
                    )}
                  </div>
              )}

              {queryResult && "results" in queryResult && (
                  <div className="space-y-4">
                    <div className="text-sm font-medium text-slate-700">Chunks recuperados</div>
                    {queryResult.results.map((item, idx) => (
                        <div key={idx} className="rounded-2xl border border-slate-200 p-4">
                          <div className="text-sm text-slate-500 mb-2">
                            {item.metadata?.source ? `Fonte: ${item.metadata.source}` : "Sem fonte"}
                            {item.metadata?.page ? ` · Página ${item.metadata.page}` : ""}
                            {item.metadata?.chunk_id !== undefined && item.metadata?.chunk_id !== null
                                ? ` · Chunk ${item.metadata.chunk_id}`
                                : ""}
                            {item.distance !== undefined && item.distance !== null
                                ? ` · Distância ${Number(item.distance).toFixed(4)}`
                                : ""}
                          </div>
                          <div className="text-sm leading-6 whitespace-pre-wrap">{item.document}</div>
                        </div>
                    ))}
                  </div>
              )}

              {ingestResult && !("answer" in ingestResult) && !("results" in ingestResult) && result && (
                  <pre className="rounded-2xl bg-slate-950 text-slate-100 p-4 overflow-auto text-xs">
                {JSON.stringify(result, null, 2)}
              </pre>
              )}
            </div>

            <div className="rounded-3xl bg-white shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold">Fluxo sugerido</h3>
              <div className="mt-3 space-y-2 text-sm text-slate-600">
                <div>
                  1. Verifique a API em <span className="font-mono">/health</span>.
                </div>
                <div>
                  2. Ingerir um documento via <span className="font-mono">/ingest-file</span> ou{" "}
                  <span className="font-mono">/ingest-text</span>.
                </div>
                <div>
                  3. Testar recuperação com <span className="font-mono">/query</span>.
                </div>
                <div>
                  4. Gerar resposta final com <span className="font-mono">/ask</span>.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
  );
}