"""Project paths and shared configuration."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAX_TOKENS = 256


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by pipeline stages."""

    root: Path
    data: Path
    raw_data: Path
    processed_data: Path
    output: Path
    figures: Path
    models: Path
    predictions: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        root = root.resolve()
        data = root / "data"
        output = root / "output"
        return cls(
            root=root,
            data=data,
            raw_data=data / "raw",
            processed_data=data / "processed",
            output=output,
            figures=output / "figures",
            models=output / "models",
            predictions=output / "predictions",
            reports=output / "reports",
        )

    def generated_directories(self) -> tuple[Path, ...]:
        return (
            self.raw_data,
            self.processed_data,
            self.figures,
            self.models,
            self.predictions,
            self.reports,
        )


def build_paths(root: Path | None = None) -> ProjectPaths:
    """Build project paths for the repository or a test root."""

    return ProjectPaths.from_root(PROJECT_ROOT if root is None else root)


DEFAULT_PATHS = build_paths()


def ensure_generated_directories(paths: ProjectPaths = DEFAULT_PATHS) -> tuple[Path, ...]:
    """Create generated data and output directories if they are missing."""

    directories = paths.generated_directories()
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


@dataclass(frozen=True)
class SourceDefinition:
    """A source that can be downloaded into `data/raw/`."""

    source_id: str
    title: str
    category: str
    source_type: str
    url: str
    raw_path: Path
    notes: str


CBRT_CONSUMER_PRICES = SourceDefinition(
    source_id="cbrt_consumer_prices",
    title="CBRT Consumer Prices",
    category="numeric",
    source_type="official_html",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Statistics/Inflation+Data/Consumer+Prices",
    raw_path=Path("numeric/cbrt_consumer_prices.html"),
    notes="Official CBRT page listing TURKSTAT CPI year-to-year and month-to-month rates.",
)

CBRT_FX_MONTH_END = SourceDefinition(
    source_id="cbrt_fx_month_end",
    title="CBRT Indicative Exchange Rates Month End Archive",
    category="numeric",
    source_type="official_xml_month_end_archive",
    url="https://www.tcmb.gov.tr/kurlar/",
    raw_path=Path("numeric/cbrt_fx_month_end"),
    notes="Official CBRT public exchange-rate XML archive sampled at each month end with business-day fallback.",
)

FRED_BRENT_OIL = SourceDefinition(
    source_id="fred_brent_oil",
    title="FRED Brent Crude Oil Price",
    category="numeric",
    source_type="fred_csv",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU",
    raw_path=Path("numeric/fred/dcoilbrenteu.csv"),
    notes="Public FRED CSV for Europe Brent crude oil spot price in USD per barrel.",
)

FRED_TURKEY_INDUSTRIAL_PRODUCTION = SourceDefinition(
    source_id="fred_turkey_industrial_production",
    title="FRED Turkey Industrial Production Growth",
    category="numeric",
    source_type="fred_csv",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=TURPRINTO01GYSAM",
    raw_path=Path("numeric/fred/turprinto01gysam.csv"),
    notes="Public FRED/OECD monthly industrial production year-over-year growth for Turkiye.",
)

FRED_TURKEY_UNEMPLOYMENT_RATE = SourceDefinition(
    source_id="fred_turkey_unemployment_rate",
    title="FRED Turkey Monthly Unemployment Rate",
    category="numeric",
    source_type="fred_csv",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=LRHUTTTTTRM156S",
    raw_path=Path("numeric/fred/lrhutttttrm156s.csv"),
    notes="Public FRED/OECD monthly seasonally adjusted unemployment rate for Turkiye.",
)

CBRT_MPC_DECISIONS = SourceDefinition(
    source_id="cbrt_mpc_decisions",
    title="CBRT MPC Meeting Decisions",
    category="text",
    source_type="official_html_listing",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%2BEN/MPC/MPC%2BMeeting%2BDecisions",
    raw_path=Path("text/cbrt_mpc_decisions.html"),
    notes="Official listing page for interest-rate press releases and MPC decisions.",
)

CBRT_MPC_SUMMARIES = SourceDefinition(
    source_id="cbrt_mpc_summaries",
    title="CBRT MPC Meeting Summaries",
    category="text",
    source_type="official_html_listing",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%2BEN/MPC/MPC%2BMeeting%2BSummaries",
    raw_path=Path("text/cbrt_mpc_summaries.html"),
    notes="Official listing page for MPC meeting summaries.",
)

SOURCE_REGISTRY: tuple[SourceDefinition, ...] = (
    CBRT_CONSUMER_PRICES,
    CBRT_FX_MONTH_END,
    FRED_BRENT_OIL,
    FRED_TURKEY_INDUSTRIAL_PRODUCTION,
    FRED_TURKEY_UNEMPLOYMENT_RATE,
    CBRT_MPC_DECISIONS,
    CBRT_MPC_SUMMARIES,
)


FRED_SERIES = {
    FRED_BRENT_OIL.source_id: ("DCOILBRENTEU", "brent_oil_usd", "daily"),
    FRED_TURKEY_INDUSTRIAL_PRODUCTION.source_id: (
        "TURPRINTO01GYSAM",
        "turkey_industrial_production_yoy_sa",
        "monthly",
    ),
    FRED_TURKEY_UNEMPLOYMENT_RATE.source_id: ("LRHUTTTTTRM156S", "turkey_unemployment_rate_sa", "monthly"),
}

CBRT_BASE_URL = "https://www.tcmb.gov.tr"
CBRT_FX_ARCHIVE_BASE_URL = "https://www.tcmb.gov.tr/kurlar"
FX_START_MONTH = date(2005, 1, 1)
ANNOUNCEMENT_CODE_PATTERN = re.compile(r"ANO(?P<year>\d{4})-(?P<number>\d{2})")
DATE_IN_TITLE_PATTERN = re.compile(r"(?P<day>\d{1,2})[/.](?P<month>\d{1,2})[/.](?P<year>\d{4})")
ENGLISH_DATE_PATTERN = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December) "
    r"(?P<day>\d{1,2}), (?P<year>\d{4})\b"
)
DAY_FIRST_ENGLISH_DATE_PATTERN = re.compile(
    r"\b(?P<day>\d{1,2}) "
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December) "
    r"(?P<year>\d{4})\b"
)
MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def sources_by_category(category: str) -> tuple[SourceDefinition, ...]:
    """Return sources for one registry category."""

    return tuple(source for source in SOURCE_REGISTRY if source.category == category)


def source_by_id(source_id: str) -> SourceDefinition:
    """Return one source definition by id."""

    for source in SOURCE_REGISTRY:
        if source.source_id == source_id:
            return source
    raise KeyError(f"Unknown source id: {source_id}")


def normalize_cbrt_url(href: str) -> str:
    return urljoin(CBRT_BASE_URL, href)


def cbrt_document_id(source_id: str, url: str) -> str:
    match = ANNOUNCEMENT_CODE_PATTERN.search(url)
    if match:
        return f"{source_id}_{match.group(0).lower()}"
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    return f"{source_id}_{path.lower()}"


def raw_document_path(document_id: str) -> Path:
    """Return the deterministic raw HTML path for one official text document."""

    return Path("text/documents") / f"{document_id}.html"


def published_at_from_title(title: str) -> pd.Timestamp | pd.NaT:
    match = DATE_IN_TITLE_PATTERN.search(title)
    if not match:
        return pd.NaT
    return pd.Timestamp(year=int(match.group("year")), month=int(match.group("month")), day=int(match.group("day")))


def published_at_from_body_text(text: str) -> pd.Timestamp | pd.NaT:
    match = ENGLISH_DATE_PATTERN.search(text) or DAY_FIRST_ENGLISH_DATE_PATTERN.search(text)
    if not match:
        return pd.NaT
    return pd.Timestamp(
        year=int(match.group("year")),
        month=MONTHS[match.group("month")],
        day=int(match.group("day")),
    )


def extract_cbrt_text_links(html: str, source: SourceDefinition) -> pd.DataFrame:
    """Extract official CBRT text document links from a listing page."""

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    expected_path = source.url.split("/EN/TCMB%2BEN/")[-1].replace("%2B", "+")

    for anchor in soup.find_all("a", href=True):
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if not title:
            continue
        url = normalize_cbrt_url(anchor["href"])
        if expected_path not in url or not ANNOUNCEMENT_CODE_PATTERN.search(url):
            continue
        document_id = cbrt_document_id(source.source_id, url)
        rows.append(
            {
                "document_id": document_id,
                "source_id": source.source_id,
                "source_type": source.source_type,
                "title": title,
                "url": url,
                "published_at": published_at_from_title(title),
                "raw_listing_path": source.raw_path.as_posix(),
                "raw_document_path": raw_document_path(document_id).as_posix(),
            }
        )

    if not rows:
        raise ValueError(f"Could not extract text links from {source.source_id}")

    documents = pd.DataFrame(rows).drop_duplicates(subset=["document_id"]).reset_index(drop=True)
    return documents.sort_values(["published_at", "document_id"], na_position="last").reset_index(drop=True)


def latest_completed_month(today: date | None = None) -> date:
    """Return the first day of the latest completed calendar month."""

    today = datetime.now(UTC).date() if today is None else today
    first_day_this_month = date(today.year, today.month, 1)
    latest_month_end = first_day_this_month - timedelta(days=1)
    return date(latest_month_end.year, latest_month_end.month, 1)


def iter_month_starts(start_month: date = FX_START_MONTH, end_month: date | None = None) -> list[date]:
    """Return month starts from `start_month` through `end_month`, inclusive."""

    end_month = latest_completed_month() if end_month is None else end_month
    month = date(start_month.year, start_month.month, 1)
    end_month = date(end_month.year, end_month.month, 1)
    months = []
    while month <= end_month:
        months.append(month)
        year = month.year + (month.month // 12)
        next_month = 1 if month.month == 12 else month.month + 1
        month = date(year, next_month, 1)
    return months


def month_end_candidates(month_start: date, fallback_days: int = 14) -> list[date]:
    """Return candidate archive dates from month-end backward."""

    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    month_end = date(month_start.year, month_start.month, last_day)
    return [month_end - timedelta(days=offset) for offset in range(fallback_days + 1)]


def cbrt_fx_url_for_date(effective_date: date) -> str:
    """Build the public CBRT XML URL for one archive date."""

    return f"{CBRT_FX_ARCHIVE_BASE_URL}/{effective_date:%Y%m}/{effective_date:%d%m%Y}.xml"


def cbrt_fx_raw_path_for_date(base_path: Path, effective_date: date) -> Path:
    """Build the local raw XML path for one archive date."""

    return base_path / f"{effective_date:%Y%m}" / f"{effective_date:%d%m%Y}.xml"
