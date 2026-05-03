import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import {
	type AgentStreamEvent,
	invalidateSignalsCache,
	startAgentRun,
	streamAgentRun,
} from '../api/client'

type NodeState = {
	status: 'idle' | 'running' | 'done' | 'error'
	text: string
	data: Record<string, unknown>
}

type RunResult = {
	decision: string | null
	feedback: string | null
	ic_tstat: number | null
	sharpe_net: number | null
}

const NODE_ORDER = ['researcher', 'coder', 'critic']
const EMPTY_NODE: NodeState = { status: 'idle', text: '', data: {} }

function badgeClass(status: NodeState['status']): string {
	if (status === 'done') return 'chip chip-success'
	if (status === 'error') return 'chip chip-danger'
	if (status === 'running') return 'chip chip-warn'
	return 'chip'
}

function ResearcherBody({ node }: { node: NodeState }) {
	if (node.status === 'idle') return <p className="node-idle">Waiting to start…</p>
	if (node.status === 'running') return <p className="node-idle">Searching arXiv for papers…</p>
	const hypothesis = typeof node.data.signal_hypothesis === 'string' ? node.data.signal_hypothesis : ''
	const ids = Array.isArray(node.data.arxiv_ids) ? (node.data.arxiv_ids as string[]) : []
	return (
		<div className="node-body">
			<div className="node-pills">
				{ids.length > 0
					? <span className="pill">{ids.length} paper{ids.length !== 1 ? 's' : ''} found</span>
					: <span className="pill">No arXiv results — using stored context</span>}
			</div>
			{hypothesis
				? <p className="node-detail">{hypothesis.length > 220 ? hypothesis.slice(0, 220) + '…' : hypothesis}</p>
				: <p className="node-idle">No hypothesis — coder will use raw theme.</p>}
		</div>
	)
}

function CoderBody({ node }: { node: NodeState }) {
	if (node.status === 'idle') return <p className="node-idle">Waiting to start…</p>
	const retryCount = typeof node.data.retry_count === 'number' ? node.data.retry_count : null
	return (
		<div className="node-body">
			<pre className="node-text">{node.text || (node.status === 'running' ? 'Generating signal code…' : '')}</pre>
			{node.status === 'done' && retryCount !== null && (
				<div className="node-pills">
					<span className="pill">attempt {retryCount}</span>
				</div>
			)}
		</div>
	)
}

function CriticBody({ node }: { node: NodeState }) {
	if (node.status === 'idle') return <p className="node-idle">Waiting to start…</p>
	if (node.status === 'running') return <p className="node-idle">Running backtest…</p>
	const decision = typeof node.data.decision === 'string' ? node.data.decision : null
	const icTstat = typeof node.data.ic_tstat === 'number' ? node.data.ic_tstat : null
	const sharpe = typeof node.data.sharpe_net === 'number' ? node.data.sharpe_net : null
	const feedback = typeof node.data.feedback === 'string' ? node.data.feedback : ''
	return (
		<div className="node-body">
			<div className="node-pills">
				{icTstat !== null && <span className="pill">IC t-stat {icTstat.toFixed(2)}</span>}
				{sharpe !== null && <span className="pill">Sharpe {sharpe.toFixed(2)}</span>}
				{decision && (
					<span className={`chip ${decision === 'accept' ? 'chip-success' : 'chip-danger'}`}>
						{decision}
					</span>
				)}
			</div>
			{feedback && (
				<p className="node-detail">{feedback.length > 220 ? feedback.slice(0, 220) + '…' : feedback}</p>
			)}
		</div>
	)
}

function NodeBody({ name, node }: { name: string; node: NodeState }) {
	if (name === 'researcher') return <ResearcherBody node={node} />
	if (name === 'coder') return <CoderBody node={node} />
	if (name === 'critic') return <CriticBody node={node} />
	return null
}

export function AgentLog({
	defaultTheme = 'earnings quality factors',
	runId: externalRunId,
	compact = false,
	onRunCreated,
	onRunComplete,
}: {
	defaultTheme?: string
	runId?: string | null
	compact?: boolean
	onRunCreated?: (runId: string) => void
	onRunComplete?: () => void
}) {
	const [theme, setTheme] = useState(defaultTheme)
	const [runId, setRunId] = useState<string | null>(externalRunId ?? null)
	const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
	const [error, setError] = useState<string | null>(null)
	const [runResult, setRunResult] = useState<RunResult | null>(null)
	const [nodes, setNodes] = useState<Record<string, NodeState>>(() => ({
		researcher: { ...EMPTY_NODE },
		coder: { ...EMPTY_NODE },
		critic: { ...EMPTY_NODE },
	}))
	const closeRef = useRef<(() => void) | null>(null)
	const onRunCompleteRef = useRef(onRunComplete)
	useEffect(() => { onRunCompleteRef.current = onRunComplete }, [onRunComplete])

	useEffect(() => {
		setRunId(externalRunId ?? null)
	}, [externalRunId])

	const orderedNodes = useMemo(
		() => NODE_ORDER.map((name) => ({ name, ...nodes[name] })),
		[nodes],
	)

	useEffect(() => {
		if (!runId) return undefined
		setStatus('running')
		setError(null)
		const close = streamAgentRun(runId, {
			onEvent: (event: AgentStreamEvent) => {
				if (event.event === 'node_start') {
					const node = String(event.node || 'coder')
					setNodes((current) => ({
						...current,
						[node]: { status: 'running', text: '', data: event.data },
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
					const d = event.data
					setRunResult({
						decision: typeof d.decision === 'string' ? d.decision : null,
						feedback: typeof d.feedback === 'string' ? d.feedback : null,
						ic_tstat: typeof d.ic_tstat === 'number' ? d.ic_tstat : null,
						sharpe_net: typeof d.sharpe_net === 'number' ? d.sharpe_net : null,
					})
					setStatus('done')
					invalidateSignalsCache()
					onRunCompleteRef.current?.()
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
		setRunResult(null)
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

	function handleReset() {
		setStatus('idle')
		setRunResult(null)
		setError(null)
		setRunId(null)
		setNodes({
			researcher: { ...EMPTY_NODE },
			coder: { ...EMPTY_NODE },
			critic: { ...EMPTY_NODE },
		})
	}

	const isRunning = status === 'running'
	const isFinished = status === 'done' || status === 'error'
	const isAccepted = runResult?.decision === 'accept'

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
					disabled={isRunning}
				/>
				{isFinished ? (
					<button className="secondary-button" type="button" onClick={handleReset}>
						New run
					</button>
				) : (
					<button
						className="primary-button"
						type="button"
						onClick={handleStart}
						disabled={isRunning}
					>
						{isRunning ? 'Running…' : 'Run Agent'}
					</button>
				)}
			</div>

			{error ? <div className="inline-alert inline-alert-error">{error}</div> : null}

			{runResult ? (
				<div className={`run-result ${isAccepted ? 'run-result-accept' : 'run-result-reject'}`}>
					<div className="run-result-header">
						<strong>{isAccepted ? 'Signal accepted' : 'Signal rejected'}</strong>
						<div className="run-result-pills">
							{runResult.ic_tstat !== null && (
								<span className="pill">IC t-stat {runResult.ic_tstat.toFixed(2)}</span>
							)}
							{runResult.sharpe_net !== null && (
								<span className="pill">Sharpe {runResult.sharpe_net.toFixed(2)}</span>
							)}
						</div>
					</div>
					{runResult.feedback && (
						<p className="run-result-feedback">{runResult.feedback}</p>
					)}
					{isAccepted ? (
						<p className="run-result-hint">
							The generated code is saved in the{' '}
							<Link to="/research" className="inline-link">Research log</Link>
							{' '}— copy it from there to register as a permanent signal.
						</p>
					) : (
						<p className="run-result-hint">
							Thresholds: IC t-stat ≥ 0.5 and net Sharpe ≥ 0.3.
							Try a more specific theme, or a different economic mechanism.
						</p>
					)}
				</div>
			) : null}

			<div className="node-grid">
				{orderedNodes.map((node) => (
					<article key={node.name} className={`node-card node-${node.status}`}>
						<div className="node-card-head">
							<h3>{node.name}</h3>
							<span className={badgeClass(node.status)}>{node.status}</span>
						</div>
						<NodeBody name={node.name} node={node} />
					</article>
				))}
			</div>
		</section>
	)
}
