from datetime import datetime

from recur_scan.features import (
    _aggregate_transactions,
    _parse_date,
    day_consistency_score_feature,
    get_features,
    identical_transaction_ratio_feature,
    is_deposit_feature,
    is_monthly_recurring_feature,
    is_near_periodic_interval_feature,
    is_varying_amount_recurring_feature,
    low_amount_variation_feature,
    merchant_amount_std_feature,
    merchant_interval_mean_feature,
    merchant_interval_std_feature,
    n_transactions_same_amount_feature,
    percent_transactions_same_amount_feature,
    recurrence_likelihood_feature,
    time_since_last_transaction_same_merchant_feature,
)
from recur_scan.transactions import Transaction


def test_parse_date_valid():
    assert _parse_date("2025-03-17") == datetime(2025, 3, 17)


def test_parse_date_invalid():
    assert _parse_date("invalid-date") is None


def test_aggregate_transactions():
    transactions = [Transaction(user_id="1", name="MerchantA"), Transaction(user_id="1", name="MerchantB")]
    aggregated = _aggregate_transactions(transactions)
    assert "1" in aggregated
    assert "MerchantA" in aggregated["1"]


def test_n_transactions_same_amount_feature():
    transaction = Transaction(user_id="1", amount=100.0)
    all_transactions = [Transaction(user_id="1", amount=100.0), Transaction(user_id="2", amount=100.0)]
    amount_counts = {100.0: 2}
    assert n_transactions_same_amount_feature(transaction, all_transactions, amount_counts) == 2


def test_percent_transactions_same_amount_feature():
    transaction = Transaction(user_id="1", amount=100.0)
    all_transactions = [Transaction(user_id="1", amount=100.0), Transaction(user_id="2", amount=200.0)]
    amount_counts = {100.0: 1, 200.0: 1}
    assert percent_transactions_same_amount_feature(transaction, all_transactions, amount_counts) == 0.5


def test_identical_transaction_ratio_feature():
    transaction = Transaction(user_id="1", amount=100.0)
    all_transactions = [Transaction(user_id="1", amount=100.0), Transaction(user_id="1", amount=100.0)]
    merchant_trans = [Transaction(user_id="1", amount=100.0), Transaction(user_id="1", amount=200.0)]
    assert identical_transaction_ratio_feature(transaction, all_transactions, merchant_trans) == 0.5


def test_is_monthly_recurring_feature():
    merchant_trans = [
        Transaction(user_id="1", date="2025-01-01"),
        Transaction(user_id="1", date="2025-01-15"),
        Transaction(user_id="1", date="2025-01-30"),
    ]
    assert is_monthly_recurring_feature(merchant_trans) == 1


def test_recurrence_likelihood_feature():
    merchant_trans = [Transaction(user_id="1", amount=100.0)] * 5
    interval_stats = {"mean": 30, "std": 5}
    amount_stats = {"mean": 100.0, "std": 10.0}
    assert 0 <= recurrence_likelihood_feature(merchant_trans, interval_stats, amount_stats) <= 1


def test_is_varying_amount_recurring_feature():
    interval_stats = {"mean": 30, "std": 5}
    amount_stats = {"mean": 100.0, "std": 15.0}
    assert is_varying_amount_recurring_feature(interval_stats, amount_stats) == 1


def test_day_consistency_score_feature():
    merchant_trans = [
        Transaction(user_id="1", date="2025-03-01"),
        Transaction(user_id="1", date="2025-03-08"),
        Transaction(user_id="1", date="2025-03-15"),
    ]
    assert 0 <= day_consistency_score_feature(merchant_trans) <= 1


def test_is_near_periodic_interval_feature():
    interval_stats = {"mean": 30, "std": 5}
    assert is_near_periodic_interval_feature(interval_stats) == 1


def test_merchant_amount_std_feature():
    amount_stats = {"mean": 100.0, "std": 10.0}
    assert merchant_amount_std_feature(amount_stats) == 0.1


def test_merchant_interval_std_feature():
    interval_stats = {"mean": 30, "std": 5}
    assert merchant_interval_std_feature(interval_stats) == 5


def test_merchant_interval_mean_feature():
    interval_stats = {"mean": 30, "std": 5}
    assert merchant_interval_mean_feature(interval_stats) == 30


def test_time_since_last_transaction_same_merchant_feature():
    parsed_dates = [datetime(2025, 3, 1), datetime(2025, 3, 15)]
    assert time_since_last_transaction_same_merchant_feature(parsed_dates) == 14.0


def test_is_deposit_feature():
    transaction = Transaction(user_id="1", amount=100.0)
    merchant_trans = [Transaction(user_id="1", amount=100.0)] * 4
    assert is_deposit_feature(transaction, merchant_trans) == 1


def test_low_amount_variation_feature():
    amount_stats = {"mean": 100.0, "std": 5.0}
    assert low_amount_variation_feature(amount_stats) == 1


def test_get_features():
    transaction = Transaction(user_id="1", amount=100.0, date="2025-03-17")
    all_transactions = [Transaction(user_id="1", amount=100.0)]
    features = get_features(transaction, all_transactions)
    assert isinstance(features, dict)
