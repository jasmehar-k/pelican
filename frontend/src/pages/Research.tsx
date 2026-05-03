import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import type { AgentRunLineage } from '../api/client'
import { getRunResearch, listResearchLog } from '../api/client'

function snippet(value?: string | null, length = 120): string {
  const text = (value || '').trim()
  if (!text) return 'No hypothesis recorded.'
  return text.length > length ? `${text.slice(0, length)}…` : text
}

export default function ResearchPage() {
  const [rows, setRows] = useState<AgentRunLineage[]>([])
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [detail, setDetail] = useState<AgentRunLineage | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    void listResearchLog()
      .then((entries) => {
        setRows(entries)
        setSelectedRun(entries[0]?.run_id ?? null)
      })
      .catch(() => setRows([]))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!selectedRun) {
      setDetail(null)
      return
    }
    void getRunResearch(selectedRun)
      .then(setDetail)
      .catch(() => setDetail(rows.find((row) => row.run_id === selectedRun) ?? null))
  }, [rows, selectedRun])

  return (
    <main className="page-grid page-grid-research">
      <section className="panel">
        <div className="panel-header">
          <div>
            <div className="eyebrow">Research log</div>
            <h2>Papers → hypotheses → results</h2>
            <p>Each row traces a run from arXiv context through the generated signal and backtest result. Accepted runs include the generated code.</p>
          </div>
        </div>

        {loading ? (
          <div className="empty-state"><p className="muted-text">Loading…</p></div>
        ) : rows.length === 0 ? (
          <div className="empty-state">
            <p>No research runs yet.</p>
            <p className="muted-text">Start an agent run from the dashboard. Completed runs — accepted or rejected — appear here automatically.</p>
            <Link to="/" className="secondary-button" style={{ marginTop: '12px', display: 'inline-flex' }}>Go to dashboard →</Link>
          </div>
        ) : (
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Theme</th>
                  <th>Papers</th>
                  <th>Hypothesis</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.run_id}
                    onClick={() => setSelectedRun(row.run_id)}
                    style={{ cursor: 'pointer' }}
                    className={selectedRun === row.run_id ? 'row-selected' : ''}
                  >
                    <td>{row.run_id.slice(0, 8)}</td>
                    <td>{row.theme}</td>
                    <td>{row.papers.length || row.arxiv_ids.length || '—'}</td>
                    <td>{snippet(row.signal_hypothesis, 80)}</td>
                    <td>
                      <span className={`chip ${row.decision === 'accept' ? 'chip-success' : row.decision === 'reject' ? 'chip-danger' : ''}`}>
                        {row.decision ?? 'pending'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {detail ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Run detail</div>
              <h2>{detail.theme}</h2>
            </div>
            <span className={`chip ${detail.decision === 'accept' ? 'chip-success' : detail.decision === 'reject' ? 'chip-danger' : ''}`}>
              {detail.decision ?? 'pending'}
            </span>
          </div>

          {detail.signal_hypothesis && (
            <p style={{ marginTop: 0 }}>{snippet(detail.signal_hypothesis, 320)}</p>
          )}

          <div className="detail-grid">
            <div>
              <div className="eyebrow" style={{ marginBottom: '12px' }}>Papers</div>
              <div className="paper-list">
                {detail.papers.length > 0 ? (
                  detail.papers.map((paper) => (
                    <div key={paper.arxiv_id ?? paper.title} className="paper-card">
                      <strong>{paper.title ?? paper.arxiv_id ?? 'Untitled paper'}</strong>
                      <span>{paper.authors.join(', ') || 'Unknown authors'}</span>
                      {paper.abstract && <p>{snippet(paper.abstract, 160)}</p>}
                    </div>
                  ))
                ) : (
                  <div className="muted-text">No papers stored for this run.</div>
                )}
              </div>
            </div>

            <div>
              <div className="eyebrow" style={{ marginBottom: '12px' }}>Backtest result</div>
              <div className="stack-list">
                <div className="stack-item">
                  <span>IC t-stat</span>
                  <strong>{detail.ic_tstat?.toFixed(2) ?? 'n/a'}</strong>
                </div>
                <div className="stack-item">
                  <span>Sharpe (net)</span>
                  <strong>{detail.sharpe_net?.toFixed(2) ?? 'n/a'}</strong>
                </div>
                <div className="stack-item">
                  <span>Retries</span>
                  <strong>{detail.retry_count}</strong>
                </div>
                <div className="stack-item">
                  <span>Feedback</span>
                  <strong style={{ fontSize: '0.85rem', fontWeight: 400 }}>{snippet(detail.feedback, 160)}</strong>
                </div>
              </div>

              {detail.decision === 'accept' && detail.generated_code && (
                <div style={{ marginTop: '16px' }}>
                  <div className="eyebrow" style={{ marginBottom: '8px' }}>Generated code</div>
                  <pre className="node-text" style={{ maxHeight: '320px', overflow: 'auto', fontSize: '0.78rem' }}>
                    {detail.generated_code}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </section>
      ) : rows.length > 0 ? (
        <section className="panel">
          <div className="empty-state">
            <p className="muted-text">Select a run from the table to see its detail.</p>
          </div>
        </section>
      ) : null}
    </main>
  )
}
