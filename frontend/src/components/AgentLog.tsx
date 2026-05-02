import { useEffect, useMemo, useRef, useState } from 'react'

import {
	AgentStreamEvent,
	invalidateSignalsCache,
	startAgentRun,
	streamAgentRun,
} from '../api/client'

type NodeState = {
	status: 'idle' | 'running' | 'done' | 'error'
	text: string
	data: Record<string, unknown>
}

const NODE_ORDER = ['researcher', 'coder', 'critic']

const EMPTY_NODE: NodeState = { status: 'idle', text: '', data: {} }

function badgeClass(status: NodeState['status']): string {
	if (status === 'done') return 'chip chip-success'
	if (status === 'error') return 'chip chip-danger'
	if (status === 'running') return 'chip chip-warn'
	return 'chip'
}

export function AgentLog({
	defaultTheme = 'earnings quality factors',
	runId: externalRunId,
	compact = false,
	onRunCreated,
}: {
	defaultTheme?: string
	runId?: string | null
	compact?: boolean
	onRunCreated?: (runId: string) => void
}) {
	const [theme, setTheme] = useState(defaultTheme)
	const [runId, setRunId] = useState<string | null>(externalRunId ?? null)
	const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
	const [error, setError] = useState<string | null>(null)
	const [nodes, setNodes] = useState<Record<string, NodeState>>(() => ({
		researcher: { ...EMPTY_NODE },
		coder: { ...EMPTY_NODE },
		critic: { ...EMPTY_NODE },
	}))
	const [liveEvent, setLiveEvent] = useState<AgentStreamEvent | null>(null)
	const closeRef = useRef<(() => void) | null>(null)

	useEffect(() => {
		setRunId(externalRunId ?? null)
	}, [externalRunId])

	const orderedNodes = useMemo(
		() => NODE_ORDER.map((name) => ({ name, ...nodes[name] })),
		[nodes],
	)

	useEffect(() => {
		if (!runId) {
			return undefined
		}
		setStatus('running')
		setError(null)
		const close = streamAgentRun(runId, {
			onEvent: (event) => {
				setLiveEvent(event)
				if (event.event === 'node_start') {
					const node = String(event.node || 'coder')
					setNodes((current) => ({
						...current,
						[node]: {
							status: 'running',
							text: '',
							data: event.data,
						},
					}))
				}
				if (event.event === 'llm_token') {
					const token = String(event.data.token || '')
					setNodes((current) => ({
						...current,
						coder: {
							status: 'running',
							text: `${current.coder.text}${token}`,
							data: current.coder.data,
						},
					}))
				}
				if (event.event === 'node_complete') {
					const node = String(event.node || 'coder')
					setNodes((current) => ({
						...current,
						[node]: {
							status: 'done',
							text: current[node]?.text || '',
							data: event.data,
						},
					}))
				}
				if (event.event === 'run_complete') {
					setStatus('done')
					invalidateSignalsCache()
					// Close the EventSource so onerror doesn't fire when the
					// server ends the connection and override our 'done' status.
					closeRef.current?.()
				}
				if (event.event === 'run_error') {
					setStatus('error')
					setError(String(event.data.error || 'run failed'))
					closeRef.current?.()
				}
			},
			onError: (streamError) => {
				setStatus('error')
				setError(streamError.message)
			},
		})
		closeRef.current = close
		return close
	}, [runId])

	async function handleStart() {
		setStatus('running')
		setError(null)
		setNodes({
			researcher: { ...EMPTY_NODE },
			coder: { ...EMPTY_NODE },
			critic: { ...EMPTY_NODE },
		})
		try {
			const response = await startAgentRun({ theme, with_researcher: true })
			setRunId(response.run_id)
			onRunCreated?.(response.run_id)
		} catch (startError) {
			setStatus('error')
			setError(startError instanceof Error ? startError.message : 'failed to start run')
		}
	}

	return (
		<section className={`panel agent-log ${compact ? 'agent-log-compact' : ''}`}>
			<div className="panel-header">
				<div>
					<div className="eyebrow">Agent</div>
					<h2>Run a research pipeline</h2>
				</div>
				<span className={badgeClass(status)}>{status}</span>
			</div>

			<div className="agent-controls">
				<input
					value={theme}
					onChange={(event) => setTheme(event.target.value)}
					placeholder="Describe the theme to research"
				/>
				<button className="primary-button" type="button" onClick={handleStart}>
					Run Agent
				</button>
			</div>

			{error ? <div className="inline-alert inline-alert-error">{error}</div> : null}

			<div className="node-grid">
				{orderedNodes.map((node) => (
					<article key={node.name} className={`node-card node-${node.status}`}>
						<div className="node-card-head">
							<h3>{node.name}</h3>
							<span className={badgeClass(node.status)}>{node.status}</span>
						</div>
						<pre className="node-text">{node.text || 'Waiting for events…'}</pre>
						{Object.keys(node.data).length > 0 ? (
							<div className="node-meta">
								{Object.entries(node.data)
									.filter(([, value]) => value !== null && value !== undefined)
									.slice(0, 3)
									.map(([key, value]) => (
										<span key={key} className="pill">
											{key}: {String(value)}
										</span>
									))}
							</div>
						) : null}
					</article>
				))}
			</div>

			{liveEvent ? (
				<div className="muted-row">
					<span className="pill">event: {liveEvent.event}</span>
					{runId ? <span className="pill">run: {runId.slice(0, 8)}</span> : null}
				</div>
			) : null}
		</section>
	)
}
