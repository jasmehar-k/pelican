import { useMemo, useState } from 'react'

import type { SignalSummary } from '../api/client'

type SortKey = 'name' | 'ic' | 'sharpe' | 'turnover' | 'coverage'

function metric(signal: SignalSummary, key: SortKey): number {
	const stats = signal.stats
	if (key === 'ic') return stats?.ic_mean ?? Number.NEGATIVE_INFINITY
	if (key === 'sharpe') return stats?.sharpe_net ?? Number.NEGATIVE_INFINITY
	if (key === 'turnover') return stats?.avg_turnover ?? Number.POSITIVE_INFINITY
	if (key === 'coverage') return stats?.avg_universe_size ?? 0
	return 0
}

function statusFor(signal: SignalSummary): string {
	if (signal.error) return 'rejected'
	if (signal.stats) return 'accepted'
	return 'pending'
}

export function SignalBrowser({
	signals,
	onSignalSelect,
}: {
	signals: SignalSummary[]
	onSignalSelect?: (signal: SignalSummary) => void
}) {
	const [query, setQuery] = useState('')
	const [sortKey, setSortKey] = useState<SortKey>('ic')
	const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
	const [statusFilter, setStatusFilter] = useState<'all' | 'accepted' | 'pending' | 'rejected'>('all')

	const rows = useMemo(() => {
		const filtered = signals.filter((signal) => {
			const haystack = `${signal.name} ${signal.description}`.toLowerCase()
			const matchesQuery = haystack.includes(query.toLowerCase())
			const status = statusFor(signal)
			const matchesStatus = statusFilter === 'all' || statusFilter === status
			return matchesQuery && matchesStatus
		})

		return filtered.sort((left, right) => {
			const leftStatus = statusFor(left)
			const rightStatus = statusFor(right)
			if (sortKey === 'name') {
				return sortDir === 'asc'
					? left.name.localeCompare(right.name)
					: right.name.localeCompare(left.name)
			}
			const leftValue = metric(left, sortKey)
			const rightValue = metric(right, sortKey)
			const delta = leftValue - rightValue
			if (delta !== 0) {
				return sortDir === 'asc' ? delta : -delta
			}
			return leftStatus.localeCompare(rightStatus)
		})
	}, [query, signals, sortDir, sortKey, statusFilter])

	function toggleSort(key: SortKey) {
		if (sortKey === key) {
			setSortDir((current) => (current === 'asc' ? 'desc' : 'asc'))
			return
		}
		setSortKey(key)
		setSortDir(key === 'turnover' ? 'asc' : 'desc')
	}

	return (
		<section className="panel">
			<div className="panel-header">
				<div>
					<div className="eyebrow">Signal browser</div>
					<h2>All discovered signals</h2>
					<p>Sort by IC, Sharpe, or turnover and open any row for the full tearsheet.</p>
				</div>
				<div className="toolbar-inline">
					<input
						value={query}
						onChange={(event) => setQuery(event.target.value)}
						placeholder="Search signals"
					/>
					<select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)}>
						<option value="all">All</option>
						<option value="accepted">Accepted</option>
						<option value="pending">Pending</option>
						<option value="rejected">Rejected</option>
					</select>
				</div>
			</div>

			<div className="table-shell">
				<table>
					<thead>
						<tr>
							<th>
								<button type="button" className="table-sort" onClick={() => toggleSort('name')}>Signal</button>
							</th>
							<th>
								<button type="button" className="table-sort" onClick={() => toggleSort('ic')}>IC</button>
							</th>
							<th>
								<button type="button" className="table-sort" onClick={() => toggleSort('sharpe')}>Sharpe</button>
							</th>
							<th>
								<button type="button" className="table-sort" onClick={() => toggleSort('turnover')}>Turnover</button>
							</th>
							<th>
								<button type="button" className="table-sort" onClick={() => toggleSort('coverage')}>Coverage</button>
							</th>
							<th>Status</th>
						</tr>
					</thead>
					<tbody>
						{rows.map((signal) => (
							<tr key={signal.name} onClick={() => onSignalSelect?.(signal)}>
								<td>
									<div className="signal-name-cell">
										<strong>{signal.name}</strong>
										<span>{signal.description}</span>
									</div>
								</td>
								<td>
									<span className="metric-value">{signal.stats?.ic_mean?.toFixed(3) ?? 'n/a'}</span>
									<span className="metric-sub">t={signal.stats?.ic_tstat?.toFixed(2) ?? 'n/a'}</span>
								</td>
								<td>
									<span className="metric-value">{signal.stats?.sharpe_net?.toFixed(2) ?? 'n/a'}</span>
								</td>
								<td>
									<span className="metric-value">{signal.stats?.avg_turnover !== null && signal.stats?.avg_turnover !== undefined ? `${(signal.stats.avg_turnover * 100).toFixed(1)}%` : 'n/a'}</span>
								</td>
								<td>
									<span className="metric-value">{signal.stats?.avg_universe_size?.toFixed(0) ?? 'n/a'}</span>
								</td>
								<td>
									<span className={`chip ${statusFor(signal) === 'accepted' ? 'chip-success' : statusFor(signal) === 'rejected' ? 'chip-danger' : ''}`}>
										{statusFor(signal)}
									</span>
								</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>
		</section>
	)
}
