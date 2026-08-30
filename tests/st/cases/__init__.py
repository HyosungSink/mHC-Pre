"""Separate Coverage and Correctness case catalogs; Mock lives in mock.json."""

from .coverage import INVALID_CASES as COVERAGE_INVALID_CASES
from .coverage import VALID_CASES as COVERAGE_VALID_CASES
from .correctness import CASES as CORRECTNESS_CASES

VALID_CASES = COVERAGE_VALID_CASES + CORRECTNESS_CASES
INVALID_CASES = COVERAGE_INVALID_CASES
CASE_GROUPS = {
    "coverage": COVERAGE_VALID_CASES,
    "correctness": CORRECTNESS_CASES,
}

_names = [case.name for case in VALID_CASES + INVALID_CASES]
if len(_names) != len(set(_names)):
    raise ValueError("duplicate mHcPre case name across categories")


def accepted_cases(suite: str = "full"):
    if suite == "full":
        return VALID_CASES
    try:
        return CASE_GROUPS[suite]
    except KeyError as error:
        raise ValueError(f"unknown case suite: {suite}") from error


def rejected_cases(suite: str = "full"):
    if suite in ("coverage", "full"):
        return COVERAGE_INVALID_CASES
    if suite == "correctness":
        return ()
    raise ValueError(f"unknown case suite: {suite}")
