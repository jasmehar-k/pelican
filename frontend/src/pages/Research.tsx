import { useEffect, useState } from 'react'

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

  useEffect(() => {
    void listResearchLog()
      .then((entries) => {
        setRows(entries)
        setSelectedRun(entries[0]?.run_id ?? null)
      })
      .catch(() => setRows([]))
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
            <p>Each row traces a research run from arXiv context through the generated signal and backtest result.</p>
          </div>
        </div>

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
                <tr key={row.run_id} onClick={() => setSelectedRun(row.run_id)}>
                  <td>{row.run_id.slice(0, 8)}</td>
                  <td>{row.theme}</td>
                  <td>{row.papers.length || row.arxiv_ids.length}</td>
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
      </section>

      {detail ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Run detail</div>
              <h2>{detail.theme}</h2>
              <p>{snippet(detail.signal_hypothesis, 240)}</p>
            </div>
          </div>

          <div className="detail-grid">
            <div>
              <div className="eyebrow">Papers</div>
              <div className="paper-list">
                {detail.papers.length > 0 ? (
                  detail.papers.map((paper) => (
                    <div key={paper.arxiv_id ?? paper.title} className="paper-card">
                      <strong>{paper.title ?? paper.arxiv_id ?? 'Untitled paper'}</strong>
                      <span>{paper.authors.join(', ') || 'Unknown authors'}</span>
                      <p>{snippet(paper.abstract, 160)}</p>
                    </div>
                  ))
                ) : (
                  <div className="muted-text">No persisted paper payload yet.</div>
                )}
              </div>
            </div>
            <div>
              <div className="eyebrow">Result</div>
              <div className="stack-list">
                <div className="stack-item">
                  <span>Decision</span>
                  <strong>{detail.decision ?? 'pending'}</strong>
                </div>
                <div className="stack-item">
                  <span>IC t-stat</span>
                  <strong>{detail.ic_tstat?.toFixed(2) ?? 'n/a'}</strong>
                </div>
                <div className="stack-item">
                  <span>Sharpe</span>
                  <strong>{detail.sharpe_net?.toFixed(2) ?? 'n/a'}</strong>
                </div>
                <div className="stack-item">
                  <span>Feedback</span>
                  <strong>{snippet(detail.feedback, 120)}</strong>
                </div>
              </div>
            </div>
          </div>
        </section>
      ) : null}
    </main>
  )
}