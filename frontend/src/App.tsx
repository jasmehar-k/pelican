import { NavLink, Navigate, Route, Routes } from 'react-router-dom'

import pelicanLogo from './assets/pelican.png'
import DashboardPage from './pages/Dashboard'
import FactorTearsheetPage from './pages/FactorTearsheet'
import PortfolioPage from './pages/Portfolio'
import ResearchPage from './pages/Research'
import SignalsPage from './pages/Signals'

const navItems = [
	{ to: '/', label: 'Dashboard' },
	{ to: '/signals', label: 'Signals' },
	{ to: '/portfolio', label: 'Portfolio' },
	{ to: '/research', label: 'Research log' },
]

export default function App() {
	return (
		<div className="app-shell">
			<header className="topbar">
				<NavLink to="/" className="topbar-brand">
					<img src={pelicanLogo} alt="Pelican" className="topbar-logo" />
				</NavLink>
				<nav className="nav">
					{navItems.map((item) => (
						<NavLink
							key={item.to}
							to={item.to}
							className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
							end={item.to === '/'}
						>
							{item.label}
						</NavLink>
					))}
				</nav>
			</header>

			<Routes>
				<Route path="/" element={<DashboardPage />} />
				<Route path="/signals" element={<SignalsPage />} />
				<Route path="/signals/:signalName" element={<FactorTearsheetPage />} />
				<Route path="/portfolio" element={<PortfolioPage />} />
				<Route path="/research" element={<ResearchPage />} />
				<Route path="*" element={<Navigate to="/" replace />} />
			</Routes>
		</div>
	)
}
