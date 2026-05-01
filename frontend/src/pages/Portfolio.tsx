import { useEffect, useMemo, useState } from 'react'

import {
	Bar,
	BarChart,
	CartesianGrid,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from 'recharts'

import type { PortfolioOptimizeResponse, SignalSummary, SignalTearsheet } from '../api/client'
import { getTearsheet, listSignals, optimizePortfolio } from '../api/client'

type BasketPoint = {
	date: string
	equity: number
}

function cumulative(values: number[]): BasketPoint[] {
	let equity = 1
	return values.map((value, index) => {
		equity *= 1 + value
		return { date: String(index), equity: equity - 1 }
	})
}

function buildBasketCurve(tearsheets: SignalTearsheet[]): BasketPoint[] {
	const byDate = new Map<string, number[]>()
	tearsheets.forEach((sheet) => {
		sheet.period_returns.forEach((row) => {
			const current = byDate.get(row.date) ?? []
			current.push(row.ls_net ?? row.ls_gross ?? 0)
			byDate.set(row.date, current)
		})
	})
	const dates = Array.from(byDate.keys()).sort()
	const returns = dates.map((date) => {
		const values = byDate.get(date) ?? []
		return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
	})
	const curve = cumulative(returns)
	return dates.map((date, index) => ({ date, equity: curve[index]?.equity ?? 0 }))
}

export default function PortfolioPage() {
	const [signals, setSignals] = useState<SignalSummary[]>([])
	const [selectedSignals, setSelectedSignals] = useState<string[]>([])
	const [result, setResult] = useState<PortfolioOptimizeResponse | null>(null)
	const [basket, setBasket] = useState<BasketPoint[]>([])
	const [busy, setBusy] = useState(false)

	useEffect(() => {
		void listSignals().then((rows) => {
			const sorted = [...rows].sort((left, right) => (right.stats?.ic_mean ?? 0) - (left.stats?.ic_mean ?? 0))
			setSignals(sorted)
			setSelectedSignals(sorted.slice(0, 3).map((row) => row.name))
		})
	}, [])

	useEffect(() => {
		const active = signals.filter((signal) => selectedSignals.includes(signal.name))
		if (active.length === 0) {
			setBasket([])
			return
		}
		void Promise.all(active.map((signal) => getTearsheet(signal.name)))
			.then((sheets) => setBasket(buildBasketCurve(sheets)))
			.catch(() => setBasket([]))
	}, [selectedSignals, signals])

	const topChoices = useMemo(() => signals.slice(0, 8), [signals])

	async function optimizeCurrentPortfolio() {
		if (selectedSignals.length === 0) {
			return
		}
		setBusy(true)
		try {
			const response = await optimizePortfolio({
				signals: selectedSignals,
				objective: 'max_sharpe',
				method: 'ic_weighted',
			})
			setResult(response)
		} finally {
			setBusy(false)
		}
	}

	const weights = result?.positions ?? []
	const risk = result?.risk_decomposition ?? null

	return (
		<main className="page-grid page-grid-portfolio">
			<section className="panel">
				<div className="panel-header">
					<div>
						<div className="eyebrow">Portfolio</div>
						<h2>Selectable signal basket</h2>
						<p>Pick a signal set, optimize it, and inspect the weight and risk footprint.</p>
					</div>
					<button className="primary-button" type="button" onClick={optimizeCurrentPortfolio} disabled={busy || selectedSignals.length === 0}>
						{busy ? 'Optimizing…' : 'Optimize portfolio'}
					</button>
				</div>

				<div className="choice-grid">
					{topChoices.map((signal) => {
						const active = selectedSignals.includes(signal.name)
						return (
							<button
								key={signal.name}
								type="button"
								className={`choice-card ${active ? 'active' : ''}`}
								onClick={() => {
									setSelectedSignals((current) => (
										current.includes(signal.name)
											? current.filter((item) => item !== signal.name)
											: [...current, signal.name]
									))
								}}
							>
								<strong>{signal.name}</strong>
								<span>IC {signal.stats?.ic_mean?.toFixed(3) ?? 'n/a'} · Sharpe {signal.stats?.sharpe_net?.toFixed(2) ?? 'n/a'}</span>
							</button>
						)
					})}
				</div>

				<div className="chart-card">
					<div className="chart-header">
						<div>
							<h3>Composite basket performance</h3>
							<p>Proxy cumulative return from the selected signal basket.</p>
						</div>
					</div>
					<div className="chart-wrap">
						<ResponsiveContainer width="100%" height={300}>
							<LineChart data={basket} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
								<CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
								<XAxis dataKey="date" tickFormatter={(value) => String(value).slice(0, 10)} />
								<YAxis tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
								<Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Equity']} />
								<Line type="monotone" dataKey="equity" stroke="#22c55e" strokeWidth={3} dot={false} />
							</LineChart>
						</ResponsiveContainer>
					</div>
				</div>
			</section>

			<section className="chart-grid">
				<div className="chart-card">
					<div className="chart-header">
						<div>
							<h3>Weights</h3>
							<p>Latest optimizer output.</p>
						</div>
					</div>
					<div className="chart-wrap">
						<ResponsiveContainer width="100%" height={300}>
							<BarChart data={weights} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
								<CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
								<XAxis dataKey="ticker" />
								<YAxis />
								<Tooltip formatter={(value: number) => [value.toFixed(3), 'Weight']} />
								<Bar dataKey="weight" fill="#38bdf8" radius={[6, 6, 0, 0]} />
							</BarChart>
						</ResponsiveContainer>
					</div>
				</div>

				<div className="chart-card">
					<div className="chart-header">
						<div>
							<h3>Portfolio curve</h3>
							<p>Proxy cumulative return from the selected signal basket.</p>
						</div>
					</div>
					<div className="chart-wrap">
						<ResponsiveContainer width="100%" height={300}>
							<LineChart data={basket} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
								<CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
								<XAxis dataKey="date" tickFormatter={(value) => String(value).slice(0, 10)} />
								<YAxis tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
								<Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Equity']} />
								<Line type="monotone" dataKey="equity" stroke="#22c55e" strokeWidth={3} dot={false} />
							</LineChart>
						</ResponsiveContainer>
					</div>
				</div>
			</section>

			<section className="panel">
				<div className="panel-header">
					<div>
						<div className="eyebrow">Risk attribution</div>
						<h2>Current optimizer output</h2>
					</div>
				</div>

				{result ? (
					<>
						<div className="metric-grid metric-grid-tight">
							<div className="metric-card"><div className="eyebrow">Status</div><div className="metric-number">{result.status}</div></div>
							<div className="metric-card"><div className="eyebrow">Expected Sharpe</div><div className="metric-number">{result.expected_sharpe.toFixed(2)}</div></div>
							<div className="metric-card"><div className="eyebrow">Expected return</div><div className="metric-number">{result.expected_return.toFixed(3)}</div></div>
							<div className="metric-card"><div className="eyebrow">Coverage</div><div className="metric-number">{result.alpha_coverage}</div></div>
						</div>

						{risk ? (
							<div className="detail-grid">
								<div className="stack-list">
									{risk.tickers.slice(0, 6).map((ticker, index) => (
										<div key={ticker} className="stack-item">
											<span>{ticker}</span>
											<strong>{risk.weights[index]?.toFixed(3)}</strong>
										</div>
									))}
								</div>
								<div className="stack-list">
									<div className="stack-item"><span>Systematic variance</span><strong>{risk.systematic_variance.toFixed(4)}</strong></div>
									<div className="stack-item"><span>Idiosyncratic variance</span><strong>{risk.idiosyncratic_variance.toFixed(4)}</strong></div>
									<div className="stack-item"><span>Systematic share</span><strong>{(risk.systematic_pct * 100).toFixed(1)}%</strong></div>
									<div className="stack-item"><span>Idiosyncratic share</span><strong>{(risk.idiosyncratic_pct * 100).toFixed(1)}%</strong></div>
								</div>
							</div>
						) : null}
					</>
				) : (
					<div className="muted-text">Run the optimizer to populate weights and risk attribution.</div>
				)}
			</section>
		</main>
	)
}
