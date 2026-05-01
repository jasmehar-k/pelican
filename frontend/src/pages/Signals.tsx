import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { AgentLog } from '../components/AgentLog'
import { SignalBrowser } from '../components/SignalBrowser'
import type { SignalSummary } from '../api/client'
import { listSignals } from '../api/client'

export default function SignalsPage() {
	const navigate = useNavigate()
	const [signals, setSignals] = useState<SignalSummary[]>([])

	useEffect(() => {
		void listSignals().then(setSignals).catch(() => setSignals([]))
	}, [])

	return (
		<main className="page-grid page-grid-signals">
			<section className="panel">
				<div className="panel-header">
					<div>
						<div className="eyebrow">Signal discovery</div>
						<h2>Sortable factor browser</h2>
						<p>Use the table to compare quality, then open any factor tearsheet for the full backtest.</p>
					</div>
				</div>
				<SignalBrowser signals={signals} onSignalSelect={(signal) => navigate(`/signals/${signal.name}`)} />
			</section>

			<AgentLog defaultTheme="earnings surprise sentiment" compact />
		</main>
	)
}
