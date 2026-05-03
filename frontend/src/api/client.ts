const API_BASE = '/api'

export type AgentRunRequest = {
	theme: string
	model?: string | null
	with_researcher?: boolean
	start?: string
	end?: string
}

export type AgentRunSummary = {
	run_id: string
	theme: string
	decision: string | null
	ic_tstat: number | null
	sharpe_net: number | null
	retry_count: number
	ts: string
}

export type AgentStreamEvent = {
	event: string
	node?: string | null
	data: Record<string, unknown>
	timestamp: string
}

export type ResearchPaper = {
	title?: string | null
	authors: string[]
	abstract?: string | null
	arxiv_id?: string | null
	url?: string | null
}

export type AgentRunLineage = {
	run_id: string
	ts: string
	theme: string
	arxiv_ids: string[]
	papers: ResearchPaper[]
	signal_hypothesis: string | null
	generated_code: string | null
	decision: string | null
	ic_tstat: number | null
	sharpe_net: number | null
	feedback: string | null
	retry_count: number
}

export type SignalBacktestStats = {
	ic_mean: number | null
	icir: number | null
	ic_tstat: number | null
	sharpe_gross: number | null
	sharpe_net: number | null
	max_drawdown_gross: number | null
	max_drawdown_net: number | null
	avg_turnover: number | null
	n_periods: number | null
	avg_universe_size: number | null
}

export type SignalSummary = {
	name: string
	description: string
	lookback_days: number
	requires_fundamentals: boolean
	requires_edgar: boolean
	data_deps: string[]
	edgar_data_deps: string[]
	expected_ic_range: [number, number]
	data_frequency: string
	min_score_coverage: number | null
	source: string
	stats: SignalBacktestStats | null
	error: string | null
}

export type BacktestConfigModel = {
	start: string
	end: string
	cost_bps: number
	min_universe_size: number
	min_score_coverage: number
	lookback_calendar_days: number
	quintile_n: number
}

export type BacktestPeriodRow = {
	date: string
	q1?: number | null
	q2?: number | null
	q3?: number | null
	q4?: number | null
	q5?: number | null
	ls_gross?: number | null
	ls_net?: number | null
	universe_size?: number | null
	n_scored?: number | null
	turnover?: number | null
}

export type BacktestICRow = {
	date: string
	ic?: number | null
}

export type SignalTearsheet = {
	summary: SignalSummary
	config: BacktestConfigModel
	period_returns: BacktestPeriodRow[]
	ic_series: BacktestICRow[]
}

export type PortfolioOptimizeRequest = {
	signals: string[]
	objective?: string
	method?: string
	rebalance_date?: string | null
	start?: string
	end?: string
	cost_bps?: number
	lambda_risk?: number
	max_weight?: number
	turnover_limit?: number | null
	min_universe_size?: number
	min_score_coverage?: number
	lookback_calendar_days?: number
}

export type PortfolioPosition = {
	ticker: string
	weight: number
}

export type RiskDecompositionSummary = {
	tickers: string[]
	weights: number[]
	total_variance: number
	systematic_variance: number
	idiosyncratic_variance: number
	systematic_pct: number
	idiosyncratic_pct: number
	factor_contributions: number[]
	idio_contributions: number[]
}

export type PortfolioOptimizeResponse = {
	signals: string[]
	rebalance_date: string
	objective: string
	method: string
	status: string
	expected_return: number
	expected_variance: number
	expected_sharpe: number
	positions: PortfolioPosition[]
	risk_decomposition: RiskDecompositionSummary | null
	ic_weights: Record<string, number>
	alpha_coverage: number
}

function buildUrl(path: string): string {
	return `${API_BASE}${path}`
}

async function requestJSON<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(buildUrl(path), {
		headers: {
			'Content-Type': 'application/json',
			...(init?.headers || {}),
		},
		...init,
	})
	if (!response.ok) {
		const detail = await response.text()
		throw new Error(detail || `Request failed with status ${response.status}`)
	}
	return response.json() as Promise<T>
}

let _signalsCache: SignalSummary[] | null = null

export async function listSignals(bust = false): Promise<SignalSummary[]> {
	if (!bust && _signalsCache) return _signalsCache
	const result = await requestJSON<SignalSummary[]>('/signals')
	_signalsCache = result
	return result
}

export function invalidateSignalsCache(): void {
	_signalsCache = null
}

export async function getSignal(name: string): Promise<SignalSummary> {
	return requestJSON<SignalSummary>(`/signals/${encodeURIComponent(name)}`)
}

export async function getTearsheet(name: string): Promise<SignalTearsheet> {
	return requestJSON<SignalTearsheet>(`/factors/${encodeURIComponent(name)}/tearsheet`)
}

export async function startAgentRun(payload: AgentRunRequest): Promise<{ run_id: string }> {
	return requestJSON<{ run_id: string }>('/agents/run', {
		method: 'POST',
		body: JSON.stringify(payload),
	})
}

export function streamAgentRun(
	runId: string,
	handlers: {
		onEvent?: (event: AgentStreamEvent) => void
		onError?: (error: Error) => void
	} = {},
): () => void {
	const source = new EventSource(buildUrl(`/agents/status/${encodeURIComponent(runId)}`))
	source.onmessage = (message) => {
		if (!message.data) {
			return
		}
		try {
			handlers.onEvent?.(JSON.parse(message.data) as AgentStreamEvent)
		} catch (error) {
			handlers.onError?.(error instanceof Error ? error : new Error('invalid SSE payload'))
		}
	}
	source.onerror = () => {
		handlers.onError?.(new Error('agent stream disconnected'))
	}
	return () => source.close()
}

export async function listAgentRuns(): Promise<AgentRunSummary[]> {
	return requestJSON<AgentRunSummary[]>('/agents/runs')
}

export async function getRunResearch(runId: string): Promise<AgentRunLineage> {
	return requestJSON<AgentRunLineage>(`/agents/runs/${encodeURIComponent(runId)}/research`)
}

export async function listResearchLog(): Promise<AgentRunLineage[]> {
	return requestJSON<AgentRunLineage[]>('/agents/research-log')
}

export type PortfolioBacktestRow = {
	date: string
	portfolio_return: number
	cumulative_return: number
}

export type PortfolioBacktestResponse = {
	signals: string[]
	start: string
	end: string
	n_periods: number
	sharpe_net: number | null
	max_drawdown: number | null
	total_return: number | null
	ic_weights: Record<string, number>
	equity_curve: PortfolioBacktestRow[]
}

export async function optimizePortfolio(payload: PortfolioOptimizeRequest): Promise<PortfolioOptimizeResponse> {
	return requestJSON<PortfolioOptimizeResponse>('/portfolio/optimize', {
		method: 'POST',
		body: JSON.stringify(payload),
	})
}

export async function portfolioBacktest(payload: PortfolioOptimizeRequest): Promise<PortfolioBacktestResponse> {
	return requestJSON<PortfolioBacktestResponse>('/portfolio/backtest', {
		method: 'POST',
		body: JSON.stringify(payload),
	})
}
