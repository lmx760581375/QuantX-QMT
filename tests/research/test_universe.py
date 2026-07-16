"""Point-in-time universe membership and audit tests."""

from quantx.core.research.universe import (
    PointInTimeUniverseProvider,
    UniverseAuditThresholds,
)


def test_point_in_time_membership_uses_effective_date_ranges(tmp_path):
    path = tmp_path / "all.txt"
    path.write_text(
        "SZ000001\t2020-01-01\t2022-12-31\nSH600000\t2021-01-01\t2099-12-31\nSZ000002\t2020-06-01\t2020-12-31\n",
        encoding="utf-8",
    )
    universe = PointInTimeUniverseProvider.from_qlib_instruments(path)

    assert universe.members("2020-03-01") == ("SZ000001",)
    assert universe.members("2020-07-01") == ("SZ000001", "SZ000002")
    assert universe.members("2021-07-01") == ("SH600000", "SZ000001")


def test_universe_audit_enforces_delisted_history():
    no_delisted = PointInTimeUniverseProvider.from_records([("SZ000001", "2020-01-01", "2099-12-31")])
    with_delisted = PointInTimeUniverseProvider.from_records(
        [
            ("SZ000001", "2020-01-01", "2099-12-31"),
            ("SZ000002", "2020-01-01", "2020-12-31"),
        ]
    )
    thresholds = UniverseAuditThresholds(require_delisted_history=True)

    assert no_delisted.audit(["2020-01-02", "2021-01-04"], thresholds).passed is False
    report = with_delisted.audit(["2020-01-02", "2021-01-04"], thresholds)
    assert report.passed is True
    assert report.delisted_symbol_count == 1


def test_formal_universe_evidence_reports_quote_and_status_failures():
    universe = PointInTimeUniverseProvider.from_records(
        [
            ("SZ000001", "2020-01-01", "2099-12-31"),
            ("SH600000", "2020-01-01", "2099-12-31"),
        ]
    )
    thresholds = UniverseAuditThresholds(
        require_delisted_history=False,
        require_quote_coverage=True,
        max_member_without_quote_ratio=0.2,
        require_st_status=True,
        require_suspension_status=True,
        require_historical_constituents=True,
    )

    report = universe.audit(
        ["2020-01-02", "2020-01-03"],
        thresholds,
        quoted_members_by_session={
            "2020-01-02": {"SZ000001", "SH600000"},
            "2020-01-03": {"SZ000001"},
        },
    )

    assert report.passed is False
    assert report.member_without_quote_ratio == 0.25
    assert set(report.failures) == {
        "member_without_quote_ratio",
        "st_status_not_declared",
        "suspension_status_not_declared",
        "historical_constituents_not_declared",
    }
