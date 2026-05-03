import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { AgentLog } from '../components/AgentLog'
import type { AgentRunLineage, AgentRunSummary, SignalSummary } from '../api/client'
import { listAgentRuns, listResearchLog, listSignals } from '../api/client'

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
	return (
		<article className="metric-card">
			<div className="eyebrow">{label}</div>
			<div className="metric-number">{value}</div>
			<div className="metric-detail">{detail}</div>
		</article>
	)
}

function shortSnippet(text?: string | null, length = 96): string {
	const value = (text || '').trim()
	if (!value) return 'No hypothesis recorded.'
	return value.length > length ? `${value.slice(0, length)}…` : value
}

export default function DashboardPage() {
	const [signals, setSignals] = useState<SignalSummary[]>([])
	const [runs, setRuns] = useState<AgentRunSummary[]>([])
	const [research, setResearch] = useState<AgentRunLineage[]>([])
	const [selectedRun, setSelectedRun] = useState<string | null>(null)

	useEffect(() => {
		void Promise.all([listSignals(), listAgentRuns(), listResearchLog()])
			.then(([signalRows, runRows, researchRows]) => {
				setSignals(signalRows)
				setRuns(runRows)
				setResearch(researchRows)
				setSelectedRun(researchRows[0]?.run_id ?? runRows[0]?.run_id ?? null)
			})
			.catch(() => {
				setSignals([])
				setRuns([])
				setResearch([])
			})
	}, [])

	const refreshStats = useCallback(() => {
		void Promise.all([listAgentRuns(), listResearchLog()])
			.then(([runRows, researchRows]) => {
				setRuns(runRows)
				setResearch(researchRows)
				if (researchRows.length > 0) {
					setSelectedRun(researchRows[0].run_id)
				}
			})
			.catch(() => {})
	}, [])

	const topSignals = useMemo(
		() => [...signals].sort((left, right) => (right.stats?.ic_tstat ?? -999) - (left.stats?.ic_tstat ?? -999)).slice(0, 4),
		[signals],
	)

	const selectedResearch = research.find((item) => item.run_id === selectedRun) ?? research[0] ?? null

	return (
		<main className="page-grid page-grid-dashboard">
			<section className="hero panel">
				<div>
					<h2>Signals, live agents, tearsheets, and portfolio output in one place.</h2>
					<p>
						Start a research run, inspect the ranking table, and trace each result back to the papers that inspired it.
					</p>
				</div>
				<div className="hero-actions">
					<Link className="secondary-button" to="/signals">Browse signals</Link>
					<Link className="secondary-button" to="/research">Research log</Link>
				</div>
			</section>

			<section className="metric-grid">
				<MetricCard label="Signals" value={String(signals.length)} detail="Registered and scored factors" />
				<MetricCard label="Agent runs" value={String(runs.length)} detail="Runs logged to research DB" />
				<MetricCard
					label="Best IC t-stat"
					value={topSignals[0]?.stats?.ic_tstat?.toFixed(2) ?? 'n/a'}
					detail={topSignals[0]?.name ?? 'No signal data yet'}
				/>
				<MetricCard label="Research lineage" value={String(research.length)} detail="Papers → hypotheses → results" />
			</section>

			<section className="content-grid">
				<AgentLog defaultTheme="earnings quality factors" compact onRunComplete={refreshStats} />

				<section className="panel">
					<div className="panel-header">
						<div>
							<div className="eyebrow">Top signals</div>
							<h2>Best by IC t-stat</h2>
						</div>
						<Link className="secondary-button" to="/signals">All signals</Link>
					</div>
					{topSignals.length > 0 ? (
						<div className="stack-list">
							{topSignals.map((signal) => (
								<Link key={signal.name} to={`/signals/${signal.name}`} className="stack-item stack-item-link">
									<div>
										<strong>{signal.name}</strong>
										<span>{signal.description}</span>
									</div>
									<div className="stack-item-metrics">
										<span className="pill">IC {signal.stats?.ic_mean?.toFixed(3) ?? 'n/a'}</span>
										<span className="pill">t={signal.stats?.ic_tstat?.toFixed(2) ?? 'n/a'}</span>
										<span className="pill">Sharpe {signal.stats?.sharpe_net?.toFixed(2) ?? 'n/a'}</span>
									</div>
								</Link>
							))}
						</div>
					) : (
						<div className="empty-state">
							<p>No signals scored yet.</p>
							<p className="muted-text">Run a backtest from the Signals page to populate this.</p>
						</div>
					)}
				</section>
			</section>

			<section className="panel">
				<div className="panel-header">
					<div>
						<div className="eyebrow">Research log</div>
						<h2>Recent runs</h2>
					</div>
					<Link className="secondary-button" to="/research">View all</Link>
				</div>

				{research.length > 0 ? (
					<>
						<div className="table-shell">
							<table>
								<thead>
									<tr>
										<th>Run</th>
										<th>Theme</th>
										<th>Hypothesis</th>
										<th>Result</th>
									</tr>
								</thead>
								<tbody>
									{research.slice(0, 5).map((item) => (
										<tr
											key={item.run_id}
											onClick={() => setSelectedRun(item.run_id)}
											style={{ cursor: 'pointer' }}
											className={selectedRun === item.run_id ? 'row-selected' : ''}
										>
											<td>{item.run_id.slice(0, 8)}</td>
											<td>{item.theme}</td>
											<td>{shortSnippet(item.signal_hypothesis)}</td>
											<td>
												<span className={`chip ${item.decision === 'accept' ? 'chip-success' : item.decision === 'reject' ? 'chip-danger' : ''}`}>
													{item.decision ?? 'pending'}
												</span>
											</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>

						{selectedResearch ? (
							<div className="detail-card" style={{ marginTop: '16px' }}>
								<div className="detail-grid">
									<div>
										<div className="eyebrow">Selected run</div>
										<h3 style={{ margin: '8px 0' }}>{selectedResearch.theme}</h3>
										<p style={{ margin: 0 }}>{shortSnippet(selectedResearch.signal_hypothesis, 220)}</p>
									</div>
									<div>
										<div className="eyebrow">Papers</div>
										<div className="paper-list">
											{selectedResearch.papers.length > 0 ? (
												selectedResearch.papers.slice(0, 3).map((paper) => (
													<div key={paper.arxiv_id ?? paper.title} className="paper-card">
														<strong>{paper.title ?? paper.arxiv_id ?? 'Untitled paper'}</strong>
														<span>{paper.authors.join(', ') || 'Unknown authors'}</span>
													</div>
												))
											) : (
												<div className="muted-text">No papers stored for this run.</div>
											)}
										</div>
									</div>
								</div>
							</div>
						) : null}
					</>
				) : (
					<div className="empty-state">
						<p>No research runs logged yet.</p>
						<p className="muted-text">
							Run the agent above — completed runs appear here automatically.
						</p>
					</div>
				)}
			</section>
		</main>
	)
}
