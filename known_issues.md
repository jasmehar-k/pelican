# known issues

## data layer

### yfinance returns split-adjusted prices in both auto_adjust modes (yfinance ≥ 0.2)

`yf.download(..., auto_adjust=False)` now applies yfinance's internal "price repair"
logic, so the `Close` column is split-adjusted regardless of the flag.  We rely on
`auto_adjust=True` and this behaves correctly (confirmed via AAPL 2020 4-for-1 split
visual check — no cliff, smooth continuation).

impact: none for current usage; the split adjustment is what we want.
watch for: if yfinance changes repair behaviour in a future release, re-run the
`aapl_split_check.png` sanity plot before trusting newly seeded data.

---

### incremental seeding produces stale forward returns near the boundary

`load_prices` computes `forward_return_21d` from the downloaded date window only.
if you seed 2014–2019 first and later extend to 2014–2024, the stored rows for
dec 2019 have `NaN` forward returns (correctly, given the original window).
re-seeding with the full range overwrites them correctly via `INSERT OR REPLACE`.

**always re-seed from `--start` rather than appending a new date range.**

---

### incremental seeding after a stock split creates a price discontinuity

if a split occurs after the initial seed run, the stored historical prices remain
at the pre-adjustment scale while new prices are in the post-split scale.
this creates a cliff in stored close prices and corrupts `log_return_1d` at the
boundary.

fix: re-seed the full date range after any split in the universe.  there is no
incremental update path; the seed script is designed to be idempotent.

---

### wikipedia universe changes table has community-maintained lag

entries in the historical additions/removals table on Wikipedia are sometimes
added weeks after the actual index change.  this slightly distorts point-in-time
universe membership for the most recent 1–4 weeks.

impact: negligible for backtests; relevant if using the pipeline for live paper
trading.  mitigated by sourcing constituent changes from a paid data vendor
(e.g. Norgate, Compustat) if accuracy at the boundary matters.

---

### delisted tickers that yfinance cannot resolve are silently skipped

`download_prices` logs a `WARNING no data for ticker` but continues.  tickers that
were in the S&P 500 historically but have since been delisted under a different
symbol (e.g., after mergers/acquisitions that changed the ticker) may return no
data at all.  this introduces survivorship bias for those specific names.

workaround: check `prices` row counts vs `sp500_universe` distinct tickers after
seeding and investigate any gap > 5%.

---

### zero-volume days kept in the price series

rows with `volume = 0` are not filtered (only `close <= 0` and `close = NaN` are
dropped).  zero-volume days can indicate exchange closures, trading halts, or data
errors.  signals that use volume (e.g. VWAP, turnover-based) may produce incorrect
values on these rows.

workaround: filter `volume > 0` inside signal compute functions when volume is an
input feature.

---

## tests

### `test_seed_defaults_follow_settings_env` path normalisation

`Path('./env.duckdb')` normalises to `'env.duckdb'` when converted to string
(Python drops the `./` prefix).  the test assertion was updated to reflect this.
if the settings type is ever changed from `Path` back to `str`, the assertion
will need to be updated again.
