"""Generate synthetic wealth data into a DuckDB file.

Usage:
  python scripts/generate_data.py [--out data/wealth.duckdb] [--seed 42]

Deterministic for a given seed. Synthetic data only, no real clients.

Tables:
  clients, products          dimensions
  holdings_monthly, flows    facts (amounts in USD)
  data_info                  as_of_date, seed, generator version (for audit)

AUM reconciles with flows: for each holding,
  aum_t = aum_{t-1} * (1 + monthly_return) + inflows_t - outflows_t
"""
import argparse
import calendar
import csv
import math
import random
import tempfile
from datetime import date, timedelta
from pathlib import Path

import duckdb

GENERATOR_VERSION = "1.0.0"
AS_OF = date(2026, 10, 31)
N_MONTHS = 24
N_CLIENTS = 500

MARKETS = {"Singapore": 0.55, "Hong Kong": 0.45}
# Monthly probability that a transaction is an inflow, per market
INFLOW_PROB = {"Singapore": 0.57, "Hong Kong": 0.52}

# segment: (weight, median starting AUM in USD, (min, max) products held)
SEGMENTS = {
  "Mass Affluent": (0.60, 150_000, (1, 3)),
  "HNW": (0.30, 2_000_000, (2, 5)),
  "UHNW": (0.10, 15_000_000, (4, 8)),
}

# (product_name, asset_class, annual mean return, annual volatility)
PRODUCTS = [
  ("Global Equity Fund", "Equity", 0.08, 0.16),
  ("Asia ex-Japan Equity Fund", "Equity", 0.07, 0.20),
  ("US Technology Equity Fund", "Equity", 0.11, 0.24),
  ("Singapore Dividend Equity Fund", "Equity", 0.05, 0.13),
  ("China Opportunities Fund", "Equity", 0.03, 0.26),
  ("Global Investment Grade Bond Fund", "Fixed Income", 0.04, 0.05),
  ("Asian High Yield Bond Fund", "Fixed Income", 0.05, 0.09),
  ("USD Treasury Fund", "Fixed Income", 0.035, 0.03),
  ("SGD Bond Fund", "Fixed Income", 0.03, 0.04),
  ("Balanced Portfolio", "Multi-Asset", 0.06, 0.09),
  ("Conservative Income Portfolio", "Multi-Asset", 0.045, 0.06),
  ("Growth Portfolio", "Multi-Asset", 0.075, 0.13),
  ("Sustainable Multi-Asset Portfolio", "Multi-Asset", 0.055, 0.10),
  ("Equity-Linked Note", "Structured", 0.05, 0.12),
  ("Capital Protected Note", "Structured", 0.03, 0.04),
  ("Dual Currency Investment", "Structured", 0.04, 0.06),
  ("USD Money Market Fund", "Cash", 0.04, 0.003),
  ("SGD Time Deposit", "Cash", 0.03, 0.001),
  ("HKD Time Deposit", "Cash", 0.035, 0.001),
  ("Multi-Currency Cash Account", "Cash", 0.01, 0.001),
]

# Number of extra transactions per holding per month
TXN_COUNTS = [0, 1, 2, 3]
TXN_WEIGHTS = [0.60, 0.25, 0.10, 0.05]


def month_ends(as_of: date, n: int) -> list[date]:
  """The n month-end dates ending at as_of, oldest first."""
  ends = []
  y, m = as_of.year, as_of.month
  for _ in range(n):
    ends.append(date(y, m, calendar.monthrange(y, m)[1]))
    y, m = (y, m - 1) if m > 1 else (y - 1, 12)
  return ends[::-1]


def random_date(rng: random.Random, start: date, end: date) -> date:
  return start + timedelta(days=rng.randint(0, (end - start).days))


def weighted(rng: random.Random, options: dict) -> str:
  keys = list(options)
  weights = [v[0] if isinstance(v, tuple) else v for v in options.values()]
  return rng.choices(keys, weights=weights)[0]


def generate(seed: int) -> dict[str, list[tuple]]:
  rng = random.Random(seed)
  ends = month_ends(AS_OF, N_MONTHS)
  period_start = ends[0].replace(day=1)

  products = [
    (f"P{i + 1:02d}", name, asset_class)
    for i, (name, asset_class, _, _) in enumerate(PRODUCTS)
  ]
  product_params = {
    f"P{i + 1:02d}": (mu, vol) for i, (_, _, mu, vol) in enumerate(PRODUCTS)
  }
  product_ids = list(product_params)

  clients = []
  # client_id -> (market, onboarding_date, {product_id: target starting AUM})
  books = {}
  for i in range(N_CLIENTS):
    cid = f"C{i + 1:04d}"
    segment = weighted(rng, SEGMENTS)
    market = weighted(rng, MARKETS)
    if rng.random() < 0.7:
      onboarded = random_date(rng, date(2015, 1, 1), period_start - timedelta(days=1))
    else:
      onboarded = random_date(rng, period_start, AS_OF)
    clients.append((cid, segment, market, onboarded.isoformat()))

    _, median_aum, (lo, hi) = SEGMENTS[segment]
    held = rng.sample(product_ids, rng.randint(lo, hi))
    total = rng.lognormvariate(math.log(median_aum), 0.6)
    splits = [rng.random() + 0.1 for _ in held]
    books[cid] = (market, onboarded, {
      pid: total * s / sum(splits) for pid, s in zip(held, splits)
    })

  # Holdings of pre-existing clients start at their target AUM
  state = {}
  for cid, (_, onboarded, book) in books.items():
    if onboarded < period_start:
      for pid, amt in book.items():
        state[(cid, pid)] = amt

  holdings, flows = [], []

  def add_flow(d: date, cid: str, pid: str, ftype: str, amt: float):
    flows.append((
      f"F{len(flows) + 1:07d}", d.isoformat(), cid, pid, ftype, round(amt, 2),
    ))

  for me in ends:
    ms = me.replace(day=1)
    for cid, (market, onboarded, book) in books.items():
      if onboarded > me:
        continue
      new_this_month = onboarded >= ms
      for pid, target in book.items():
        key = (cid, pid)
        if new_this_month:
          # Onboarding: initial funding is an inflow, no return this month
          add_flow(onboarded, cid, pid, "inflow", target)
          aum = target
        else:
          mu, vol = product_params[pid]
          r = rng.gauss(mu / 12, vol / math.sqrt(12))
          aum = state[key] * (1 + r)
          for _ in range(rng.choices(TXN_COUNTS, weights=TXN_WEIGHTS)[0]):
            if aum < 1:
              break
            d = random_date(rng, ms, me)
            if rng.random() < INFLOW_PROB[market]:
              amt = aum * rng.uniform(0.02, 0.15)
              add_flow(d, cid, pid, "inflow", amt)
              aum += amt
            else:
              amt = min(aum, aum * rng.uniform(0.02, 0.20))
              add_flow(d, cid, pid, "outflow", amt)
              aum -= amt
        state[key] = aum
        if aum >= 0.01:
          holdings.append((me.isoformat(), cid, pid, round(aum, 2)))

  return {
    "clients": clients,
    "products": products,
    "holdings_monthly": holdings,
    "flows": flows,
    "data_info": [(AS_OF.isoformat(), seed, GENERATOR_VERSION)],
  }


SCHEMA = {
  "clients": """
    client_id VARCHAR PRIMARY KEY,
    segment VARCHAR NOT NULL,
    market VARCHAR NOT NULL,
    onboarding_date DATE NOT NULL""",
  "products": """
    product_id VARCHAR PRIMARY KEY,
    product_name VARCHAR NOT NULL,
    asset_class VARCHAR NOT NULL""",
  "holdings_monthly": """
    month_end DATE NOT NULL,
    client_id VARCHAR NOT NULL,
    product_id VARCHAR NOT NULL,
    aum_usd DOUBLE NOT NULL,
    PRIMARY KEY (month_end, client_id, product_id)""",
  "flows": """
    flow_id VARCHAR PRIMARY KEY,
    flow_date DATE NOT NULL,
    client_id VARCHAR NOT NULL,
    product_id VARCHAR NOT NULL,
    flow_type VARCHAR NOT NULL CHECK (flow_type IN ('inflow', 'outflow')),
    amount_usd DOUBLE NOT NULL CHECK (amount_usd > 0)""",
  "data_info": """
    as_of_date DATE NOT NULL,
    seed INTEGER NOT NULL,
    generator_version VARCHAR NOT NULL""",
}


def write_duckdb(tables: dict[str, list[tuple]], out: Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  for f in (out, out.with_name(out.name + ".wal")):
    f.unlink(missing_ok=True)

  con = duckdb.connect(str(out))
  with tempfile.TemporaryDirectory() as tmp:
    for name, rows in tables.items():
      con.execute(f"CREATE TABLE {name} ({SCHEMA[name]})")
      path = Path(tmp) / f"{name}.csv"
      with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
      con.execute(f"COPY {name} FROM '{path.as_posix()}' (HEADER false)")
  con.close()


def main() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, default=repo_root / "data" / "wealth.duckdb")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  tables = generate(args.seed)
  write_duckdb(tables, args.out)

  con = duckdb.connect(str(args.out), read_only=True)
  print(f"Wrote {args.out}")
  for name in tables:
    n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"  {name:<18} {n:>8,} rows")
  aum = con.execute(
    "SELECT SUM(aum_usd) FROM holdings_monthly WHERE month_end = ?", [AS_OF]
  ).fetchone()[0]
  print(f"  Total AUM at {AS_OF}: ${aum:,.0f}")
  con.close()


if __name__ == "__main__":
  main()
