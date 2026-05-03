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

import type {
	PortfolioBacktestResponse,
	PortfolioOptimizeResponse,
	SignalSummary,
} from '../api/client'
import { listSignals, optimizePortfolio, portfolioBacktest } from '../api/client'

export default function PortfolioPage() {
	const [signals, setSignals] = useState<SignalSummary[]>([])
	const [selectedSignals, setSelectedSignals] = useState<string[]>([])
	const [result, setResult] = useState<PortfolioOptimizeResponse | null>(null)
	const [backtest, setBacktest] = useState<PortfolioBacktestResponse | null>(null)
	const [busyOptimize, setBusyOptimize] = useState(false)
	const [busyBacktest, setBusyBacktest] = useState(false)

	useEffect(() => {
		void listSignals().then((rows) => {
			const sorted = [...rows].sort((left, right) => (right.stats?.ic_mean ?? 0) - (left.stats?.ic_mean ?? 0))
			setSignals(sorted)
			setSelectedSignals(sorted.slice(0, 3).map((row) => row.name))
		})
	}, [])

	const topChoices = useMemo(() => signals.slice(0, 8), [signals])

	async function optimizeCurrentPortfolio() {
		if (selectedSignals.length === 0) return
		setBusyOptimize(true)
		try {
			const response = await optimizePortfolio({
				signals: selectedSignals,
				objective: 'max_sharpe',
				method: 'ic_weighted',
			})
			setResult(response)
		} finally {
			setBusyOptimize(false)
		}
	}

	async function runBacktest() {
		if (selectedSignals.length === 0) return
		setBusyBacktest(true)
		try {
			const response = await portfolioBacktest({
				signals: selectedSignals,
				objective: 'max_sharpe',
				method: 'ic_weighted',
			})
			setBacktest(response)
		} finally {
			setBusyBacktest(false)
		}
	}

	const weights = result?.positions ?? []
	const risk = result?.risk_decomposition ?? null
	const equityCurve = backtest?.equity_curve ?? []

	return (
		<main className="page-grid page-grid-portfolio">
			<section className="panel">
				<div className="panel-header">
					<div>
						<div className="eyebrow">Portfolio</div>
						<h2>Portfolio construction</h2>
						<p>Select signals, run a walk-forward backtest, and optimize current weights.</p>
					</div>
					<div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
						<button className="secondary-button" type="button" onClick={runBacktest} disabled={busyBacktest || selectedSignals.length === 0}>
							{busyBacktest ? 'Backtesting…' : 'Run backtest'}
						</button>
						<button className="primary-button" type="button" onClick={optimizeCurrentPortfolio} disabled={busyOptimize || selectedSignals.length === 0}>
							{busyOptimize ? 'Optimizing…' : 'Optimize portfolio'}
						</button>
					</div>
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
									setBacktest(null)
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
							<h3>IC-weighted portfolio backtest</h3>
							<p>Walk-forward net returns, weighted by each signal's historical IC.</p>
						</div>
						{backtest && (
							<div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
								{backtest.sharpe_net !== null && (
									<span className="chip chip-success">Sharpe {backtest.sharpe_net.toFixed(2)}</span>
								)}
								{backtest.max_drawdown !== null && (
									<span className="chip chip-danger">MDD {(backtest.max_drawdown * 100).toFixed(1)}%</span>
								)}
								{backtest.total_return !== null && (
									<span className="chip">Total {(backtest.total_return * 100).toFixed(1)}%</span>
								)}
								<span className="chip">{backtest.n_periods} periods</span>
							</div>
						)}
					</div>
					<div className="chart-wrap">
						{equityCurve.length > 0 ? (
							<ResponsiveContainer width="100%" height={300}>
								<LineChart data={equityCurve} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
									<CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
									<XAxis dataKey="date" tickFormatter={(value) => String(value).slice(0, 10)} />
									<YAxis tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
									<Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Cumulative return']} />
									<Line type="monotone" dataKey="cumulative_return" stroke="#22c55e" strokeWidth={3} dot={false} />
								</LineChart>
							</ResponsiveContainer>
						) : (
							<div className="empty-state">
								<p className="muted-text">Click "Run backtest" to compute the IC-weighted portfolio equity curve.</p>
							</div>
						)}
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
						{weights.length > 0 ? (
							<ResponsiveContainer width="100%" height={300}>
								<BarChart data={weights} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
									<CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
									<XAxis dataKey="ticker" />
									<YAxis />
									<Tooltip formatter={(value: number) => [value.toFixed(3), 'Weight']} />
									<Bar dataKey="weight" fill="#38bdf8" radius={[6, 6, 0, 0]} />
								</BarChart>
							</ResponsiveContainer>
						) : (
							<div className="empty-state">
								<p className="muted-text">Run the optimizer to see position weights.</p>
							</div>
						)}
					</div>
				</div>

				{!result && (
					<div className="chart-card">
						<div className="empty-state">
							<p className="muted-text">Run the optimizer to see position weights.</p>
						</div>
					</div>
				)}
			</section>

			<section className="panel">
				<div className="panel-header">
					<div>
						<div className="eyebrow">Risk attribution</div>
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
