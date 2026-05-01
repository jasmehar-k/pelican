import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { BacktestChart } from '../components/BacktestChart'
import type { SignalTearsheet } from '../api/client'
import { getTearsheet } from '../api/client'

export default function FactorTearsheetPage() {
  const { signalName = '' } = useParams()
  const [tearsheet, setTearsheet] = useState<SignalTearsheet | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!signalName) return
    void getTearsheet(signalName)
      .then((response) => setTearsheet(response))
      .catch((reason) => setError(reason instanceof Error ? reason.message : 'failed to load tearsheet'))
  }, [signalName])

  const icDecayData = tearsheet?.ic_series ?? []
  const spreadData = tearsheet?.period_returns ?? []

  const drawdownData = useMemo(() => {
    const returns = spreadData.map((row) => row.ls_net ?? row.ls_gross ?? 0)
    let equity = 1
    let peak = 1
    return spreadData.map((row, index) => {
      equity *= 1 + (returns[index] ?? 0)
      peak = Math.max(peak, equity)
      return {
        date: row.date,
        drawdown: equity / peak - 1,
      }
    })
  }, [spreadData])

  const summary = tearsheet?.summary

  return (
    <main className="page-grid page-grid-factor">
      <section className="panel">
        <div className="panel-header">
          <div>
            <div className="eyebrow">Factor tearsheet</div>
            <h2>{signalName}</h2>
            <p>{summary?.description ?? 'Loading factor description…'}</p>
          </div>
          <Link className="secondary-button" to="/signals">Back to signals</Link>
        </div>

        {error ? <div className="inline-alert inline-alert-error">{error}</div> : null}

        {summary ? (
          <div className="metric-grid metric-grid-tight">
            <div className="metric-card"><div className="eyebrow">IC mean</div><div className="metric-number">{summary.stats?.ic_mean?.toFixed(4) ?? 'n/a'}</div></div>
            <div className="metric-card"><div className="eyebrow">IC t-stat</div><div className="metric-number">{summary.stats?.ic_tstat?.toFixed(2) ?? 'n/a'}</div></div>
            <div className="metric-card"><div className="eyebrow">Sharpe</div><div className="metric-number">{summary.stats?.sharpe_net?.toFixed(2) ?? 'n/a'}</div></div>
            <div className="metric-card"><div className="eyebrow">Turnover</div><div className="metric-number">{summary.stats?.avg_turnover !== null && summary.stats?.avg_turnover !== undefined ? `${(summary.stats.avg_turnover * 100).toFixed(1)}%` : 'n/a'}</div></div>
          </div>
        ) : null}

        <BacktestChart periods={spreadData} title="Quintile spread" />
      </section>

      <section className="chart-grid">
        <div className="chart-card">
          <div className="chart-header">
            <div>
              <h3>IC decay</h3>
              <p>Per-period Spearman IC used to validate stability.</p>
            </div>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={icDecayData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
                <XAxis dataKey="date" tickFormatter={(value) => String(value).slice(0, 10)} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="ic" fill="#38bdf8" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <div>
              <h3>Drawdown</h3>
              <p>Peak-to-trough curve for the long/short spread.</p>
            </div>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={drawdownData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.18)" vertical={false} />
                <XAxis dataKey="date" tickFormatter={(value) => String(value).slice(0, 10)} />
                <YAxis tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} />
                <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Drawdown']} />
                <Area type="monotone" dataKey="drawdown" stroke="#f97316" fill="rgba(249, 115, 22, 0.25)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </main>
  )
}