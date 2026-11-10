"""Explicit observation periods for Black Marble annual and monthly composites."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CompositePeriod:
    year: int
    month: int | None = None

    def __post_init__(self):
        date(self.year, self.month if self.month is not None else 1, 1)

    @property
    def product(self) -> str:
        return "VNP46A4" if self.month is None else "VNP46A3"

    @property
    def start(self) -> date:
        return date(self.year, self.month if self.month is not None else 1, 1)

    @property
    def end(self) -> date:
        month = self.month if self.month is not None else 12
        return date(self.year, month, monthrange(self.year, month)[1])

    @property
    def day_of_year(self) -> str:
        return f"{self.start.timetuple().tm_yday:03d}"

    @property
    def label(self) -> str:
        return f"{self.year:04d}" + (f"-{self.month:02d}" if self.month is not None else "")

    def metadata(self) -> dict:
        return {"product": self.product, "collection": "002", "year": self.year,
                "month": self.month, "period_start": self.start.isoformat(),
                "period_end": self.end.isoformat(), "temporal_resolution":
                "annual" if self.month is None else "monthly"}
