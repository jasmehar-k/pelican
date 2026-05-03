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

	function refreshSignals() {
		void listSignals(true).then(setSignals).catch(() => {})
	}

	return (
		<main className="page-grid page-grid-signals">
			<SignalBrowser signals={signals} onSignalSelect={(signal) => navigate(`/signals/${signal.name}`)} />
			<AgentLog defaultTheme="earnings surprise sentiment" compact onRunComplete={refreshSignals} />
		</main>
	)
}
