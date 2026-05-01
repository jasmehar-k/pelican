import { useMemo } from 'react'

import {
	CartesianGrid,
	Legend,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from 'recharts'

import type { BacktestPeriodRow } from '../api/client'

type SeriesRow = BacktestPeriodRow & {
	q1_curve: number
	q5_curve: number
	ls_curve: number
}

function cumulative(values: Array<number | null | undefined>): number[] {
	let equity = 1
	return values.map((value) => {
		const step = typeof value === 'number' && Number.isFinite(value) ? value : 0
		equity *= 1 + step
		return equity - 1
	})
}

export function BacktestChart({
	periods,
	title = 'Equity Curve',
}: {
	periods: BacktestPeriodRow[]
	title?: string
}) {
	const data = useMemo<SeriesRow[]>(() => {
		const q1 = cumulative(periods.map((row) => row.q1 ?? 0))
		const q5 = cumulative(periods.map((row) => row.q5 ?? 0))
		const ls = cumulative(periods.map((row) => row.ls_net ?? row.ls_gross ?? 0))
		return periods.map((row, index) => ({
			...row,
			q1_curve: q1[index] ?? 0,
			q5_curve: q5[index] ?? 0,
			ls_curve: ls[index] ?? 0,
		}))
	}, [periods])

	return (
		<div className="chart-card">
			<div className="chart-header">
				<div>
					<h3>{title}</h3>
					<p>Long, short, and spread performance over time.</p>
				</div>
			</div>
			<div className="chart-wrap">
				<ResponsiveContainer width="100%" height={320}>
					<LineChart data={data} margin={{ top: 16, right: 20, left: 0, bottom: 0 }}>
						<CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
						<XAxis dataKey="date" tickFormatter={(value) => String(value).slice(0, 10)} />
						<YAxis tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
						<Tooltip
							formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, '']}
							labelFormatter={(label) => `Date: ${String(label).slice(0, 10)}`}
						/>
						<Legend />
						<Line type="monotone" dataKey="q1_curve" name="Q1" stroke="#94a3b8" strokeWidth={2} dot={false} />
						<Line type="monotone" dataKey="q5_curve" name="Q5" stroke="#f59e0b" strokeWidth={2} dot={false} />
						<Line type="monotone" dataKey="ls_curve" name="L/S" stroke="#22c55e" strokeWidth={3} dot={false} />
					</LineChart>
				</ResponsiveContainer>
			</div>
		</div>
	)
}
