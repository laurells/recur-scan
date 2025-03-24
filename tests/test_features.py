from collections import defaultdict
from datetime import datetime

import pytest

from recur_scan.features import (
    _aggregate_transactions,
    _calculate_intervals,
    _calculate_statistics,
    _parse_date,
    date_irregularity_dominance,
    day_consistency_score_feature,
    day_of_week_feature,
    identical_transaction_ratio_feature,
    interval_variability_feature,
    is_deposit_feature,
    is_monthly_recurring_feature,
    is_near_periodic_interval_feature,
    is_single_transaction_feature,
    is_varying_amount_recurring_feature,
    low_amount_variation_feature,
    merchant_amount_frequency_feature,
    merchant_amount_std_feature,
    merchant_interval_mean_feature,
    merchant_interval_std_feature,
    n_transactions_same_amount_feature,
    non_recurring_irregularity_score,
    percent_transactions_same_amount_feature,
    recurrence_likelihood_feature,
    rolling_amount_mean_feature,
    time_since_last_transaction_same_merchant_feature,
    transaction_month_feature,
    transaction_pattern_complexity,
)
from recur_scan.transactions import Transaction


# Fixtures
@pytest.fixture
def single_transaction():
    return Transaction(user_id="1", name="MerchantA", amount=100.0, date="2025-03-17")


@pytest.fixture
def recurring_transactions():
    return [
        Transaction(user_id="1", name="Netflix", amount=16.77, date="2025-01-01"),
        Transaction(user_id="1", name="Netflix", amount=16.77, date="2025-02-01"),
        Transaction(user_id="1", name="Netflix", amount=16.77, date="2025-03-01"),
    ]


@pytest.fixture
def irregular_transactions():
    return [
        Transaction(user_id="1", name="Dave", amount=55.0, date="2025-01-01"),
        Transaction(user_id="1", name="Dave", amount=55.0, date="2025-01-15"),
        Transaction(user_id="1", name="Dave", amount=55.0, date="2025-02-12"),
    ]


# Helper Tests
def test_parse_date():
    assert _parse_date("2025-03-17") == datetime(2025, 3, 17)
    assert _parse_date("invalid-date") is None


def test_aggregate_transactions(recurring_transactions):
    agg = _aggregate_transactions(recurring_transactions)
    assert len(agg["1"]["Netflix"]) == 3
    assert agg["1"]["Netflix"][0].amount == 16.77


def test_calculate_statistics(recurring_transactions):
    dates = [_parse_date(t.date) for t in recurring_transactions if _parse_date(t.date)]
    intervals = [float(i) for i in _calculate_intervals(dates)]
    stats = _calculate_statistics(intervals)
    assert stats["mean"] == pytest.approx(30.0, abs=1e-5)
    assert stats["std"] < 1.0


# Feature Tests (23/23)
def test_n_transactions_same_amount_feature(single_transaction):
    amount_counts = defaultdict(int, {100.0: 2, 200.0: 1})
    assert n_transactions_same_amount_feature(single_transaction, amount_counts) == 2
    assert n_transactions_same_amount_feature(Transaction(user_id="1", amount=300.0), amount_counts) == 0


def test_percent_transactions_same_amount_feature(single_transaction):
    all_txs = [single_transaction, Transaction(user_id="2", name="B", amount=200.0, date="2025-03-18")]
    amount_counts = defaultdict(int, {100.0: 1, 200.0: 1})
    assert percent_transactions_same_amount_feature(single_transaction, all_txs, amount_counts) == 0.5
    assert percent_transactions_same_amount_feature(single_transaction, [], amount_counts) == 0.0


def test_identical_transaction_ratio_feature(single_transaction):
    all_txs = [single_transaction, Transaction(user_id="1", name="MerchantA", amount=100.0, date="2025-03-18")]
    merchant_txs = [single_transaction, Transaction(user_id="1", name="MerchantA", amount=200.0, date="2025-03-19")]
    assert identical_transaction_ratio_feature(single_transaction, all_txs, merchant_txs) == 0.5


def test_is_monthly_recurring_feature(recurring_transactions, irregular_transactions):
    assert is_monthly_recurring_feature(recurring_transactions) == 1.0
    assert is_monthly_recurring_feature(irregular_transactions) < 0.8
    assert is_monthly_recurring_feature([]) == 0.0


def test_recurrence_likelihood_feature(recurring_transactions, irregular_transactions):
    rec_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in recurring_transactions if _parse_date(t.date)])
    ])
    irr_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in irregular_transactions if _parse_date(t.date)])
    ])
    rec_amount_stats = _calculate_statistics([t.amount for t in recurring_transactions])
    irr_amount_stats = _calculate_statistics([t.amount for t in irregular_transactions])
    rec_score = recurrence_likelihood_feature(recurring_transactions, rec_stats, rec_amount_stats)
    irr_score = recurrence_likelihood_feature(irregular_transactions, irr_stats, irr_amount_stats)
    assert rec_score > 0.9
    assert irr_score < 0.5


def test_is_varying_amount_recurring_feature():
    assert is_varying_amount_recurring_feature({"mean": 30.0, "std": 5.0}, {"mean": 100.0, "std": 0.5}) == 1
    assert is_varying_amount_recurring_feature({"mean": 60.0, "std": 50.0}, {"mean": 100.0, "std": 0.0}) == 0


def test_day_consistency_score_feature(recurring_transactions, irregular_transactions):
    assert day_consistency_score_feature(recurring_transactions) > 0.9
    assert day_consistency_score_feature(irregular_transactions) < 0.6
    assert day_consistency_score_feature([recurring_transactions[0]]) == 0.5


def test_is_near_periodic_interval_feature(recurring_transactions):
    stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in recurring_transactions if _parse_date(t.date)])
    ])
    assert is_near_periodic_interval_feature(stats) > 0.8
    assert is_near_periodic_interval_feature({"mean": 17.0, "std": 6.5}) < 0.5


def test_merchant_amount_std_feature():
    assert merchant_amount_std_feature({"mean": 100.0, "std": 10.0}) == 0.1
    assert merchant_amount_std_feature({"mean": 0.0, "std": 0.0}) == 0.0


def test_merchant_interval_std_feature():
    assert merchant_interval_std_feature({"mean": 30.0, "std": 5.0}) == 5.0
    assert merchant_interval_std_feature({"mean": 0.0, "std": 0.0}) == 30.0


def test_merchant_interval_mean_feature():
    assert merchant_interval_mean_feature({"mean": 30.0, "std": 5.0}) == 30.0
    assert merchant_interval_mean_feature({"mean": 0.0, "std": 0.0}) == 60.0


def test_time_since_last_transaction_same_merchant_feature(recurring_transactions):
    dates = [_parse_date(t.date) for t in recurring_transactions if _parse_date(t.date)]
    assert time_since_last_transaction_same_merchant_feature(dates) == pytest.approx(30.0 / 365, abs=0.01)
    assert time_since_last_transaction_same_merchant_feature([]) == 0.0


def test_is_deposit_feature(single_transaction, recurring_transactions):
    assert is_deposit_feature(single_transaction, recurring_transactions) == 1
    assert is_deposit_feature(single_transaction, [single_transaction]) == 0


def test_day_of_week_feature(single_transaction):
    assert day_of_week_feature(single_transaction) == pytest.approx(0.0 / 6)  # Monday = 0
    assert day_of_week_feature(Transaction(user_id="1", name="A", date="2025-03-23")) == pytest.approx(
        6.0 / 6
    )  # Sunday = 6


def test_transaction_month_feature(single_transaction):
    assert transaction_month_feature(single_transaction) == pytest.approx((3 - 1) / 11)  # March
    assert transaction_month_feature(Transaction(user_id="1", name="A", date="2025-01-01")) == 0.0  # January


def test_rolling_amount_mean_feature(recurring_transactions, irregular_transactions):
    assert rolling_amount_mean_feature(recurring_transactions) == pytest.approx(16.77)
    assert rolling_amount_mean_feature([irregular_transactions[0]]) == 55.0


def test_low_amount_variation_feature():
    assert low_amount_variation_feature({"mean": 100.0, "std": 5.0}) == 1  # 0.05 < 0.1
    assert low_amount_variation_feature({"mean": 100.0, "std": 20.0}) == 0  # 0.2 > 0.1


def test_is_single_transaction_feature(single_transaction, recurring_transactions):
    assert is_single_transaction_feature([single_transaction]) == 1
    assert is_single_transaction_feature(recurring_transactions) == 0


def test_interval_variability_feature():
    assert interval_variability_feature({"mean": 30.0, "std": 15.0}) == 0.5
    assert interval_variability_feature({"mean": 0.0, "std": 0.0}) == 1.0


def test_merchant_amount_frequency_feature(recurring_transactions, irregular_transactions):
    assert merchant_amount_frequency_feature(recurring_transactions) == 1  # All 16.77
    assert merchant_amount_frequency_feature(irregular_transactions) == 1  # All 55.0


def test_non_recurring_irregularity_score(recurring_transactions, irregular_transactions):
    rec_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in recurring_transactions if _parse_date(t.date)])
    ])
    irr_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in irregular_transactions if _parse_date(t.date)])
    ])
    rec_amount_stats = _calculate_statistics([t.amount for t in recurring_transactions])
    irr_amount_stats = _calculate_statistics([t.amount for t in irregular_transactions])
    rec_score = non_recurring_irregularity_score(recurring_transactions, rec_stats, rec_amount_stats)
    irr_score = non_recurring_irregularity_score(irregular_transactions, irr_stats, irr_amount_stats)
    assert rec_score < 0.2
    assert irr_score > 0.4


def test_transaction_pattern_complexity(recurring_transactions, irregular_transactions):
    rec_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in recurring_transactions if _parse_date(t.date)])
    ])
    irr_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in irregular_transactions if _parse_date(t.date)])
    ])
    rec_score = transaction_pattern_complexity(recurring_transactions, rec_stats)
    irr_score = transaction_pattern_complexity(irregular_transactions, irr_stats)
    assert rec_score < 0.2
    assert irr_score > 0.3


def test_date_irregularity_dominance(recurring_transactions, irregular_transactions):
    rec_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in recurring_transactions if _parse_date(t.date)])
    ])
    irr_stats = _calculate_statistics([
        float(i)
        for i in _calculate_intervals([_parse_date(t.date) for t in irregular_transactions if _parse_date(t.date)])
    ])
    rec_amount_stats = _calculate_statistics([t.amount for t in recurring_transactions])
    irr_amount_stats = _calculate_statistics([t.amount for t in irregular_transactions])
    rec_score = date_irregularity_dominance(recurring_transactions, rec_stats, rec_amount_stats)
    irr_score = date_irregularity_dominance(irregular_transactions, irr_stats, irr_amount_stats)
    assert rec_score < 0.3
    assert irr_score > 0.6
