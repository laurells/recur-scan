# test features_original.py

import pytest

from recur_scan.features_original import (
    detect_monthly_with_missing_entries,
    get_amount_consistency_score,
    get_combined_recurrence_score,
    get_ends_in_99,
    get_interval_consistency_score,
    get_is_albert_99_recurring,
    get_is_always_recurring,
    get_is_amazon_prime,
    get_is_insurance,
    get_is_known_non_recurring_vendor,
    get_is_known_recurring_vendor,
    get_is_phone,
    get_is_recurring_same_amount_specific_intervals,
    get_is_utility,
    get_n_transactions_days_apart,
    get_n_transactions_same_amount,
    get_n_transactions_same_day,
    get_pct_transactions_days_apart,
    get_pct_transactions_same_day,
    get_percent_transactions_same_amount,
    get_same_amount_count,
    get_transaction_z_score,
)
from recur_scan.transactions import Transaction


def test_get_n_transactions_same_amount() -> None:
    """Test that get_n_transactions_same_amount returns the correct number of transactions with the same amount."""
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=3, user_id="user1", name="name1", amount=200, date="2024-01-02"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-03"),
    ]
    assert get_n_transactions_same_amount(transactions[0], transactions) == 2
    assert get_n_transactions_same_amount(transactions[2], transactions) == 1


def test_get_percent_transactions_same_amount() -> None:
    """
    Test that get_percent_transactions_same_amount returns correct percentage.
    Tests that the function calculates the right percentage of transactions with matching amounts.
    """
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=3, user_id="user1", name="name1", amount=200, date="2024-01-02"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-03"),
    ]
    assert pytest.approx(get_percent_transactions_same_amount(transactions[0], transactions)) == 2 / 4


def test_get_ends_in_99() -> None:
    """Test that get_ends_in_99 returns True for amounts ending in 99."""
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=3, user_id="user1", name="name1", amount=200, date="2024-01-02"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-03"),
    ]
    assert not get_ends_in_99(transactions[0])
    assert get_ends_in_99(transactions[3])


def test_get_n_transactions_same_day() -> None:
    """Test that get_n_transactions_same_day returns the correct number of transactions on the same day."""
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=3, user_id="user1", name="name1", amount=200, date="2024-01-02"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-03"),
    ]
    assert get_n_transactions_same_day(transactions[0], transactions, 0) == 2
    assert get_n_transactions_same_day(transactions[0], transactions, 1) == 3
    assert get_n_transactions_same_day(transactions[2], transactions, 0) == 1


def test_get_pct_transactions_same_day() -> None:
    """Test that get_pct_transactions_same_day returns the correct percentage of transactions on the same day."""
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=3, user_id="user1", name="name1", amount=200, date="2024-01-02"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-03"),
    ]
    assert get_pct_transactions_same_day(transactions[0], transactions, 0) == 2 / 4


def test_get_n_transactions_days_apart() -> None:
    """Test get_n_transactions_days_apart."""
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=2.99, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=2.99, date="2024-01-02"),
        Transaction(id=3, user_id="user1", name="name1", amount=2.99, date="2024-01-14"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-15"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-16"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-29"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-31"),
    ]
    assert get_n_transactions_days_apart(transactions[0], transactions, 14, 0) == 2
    assert get_n_transactions_days_apart(transactions[0], transactions, 14, 1) == 4


def test_get_pct_transactions_days_apart() -> None:
    """Test get_pct_transactions_days_apart."""
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=2.99, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=2.99, date="2024-01-02"),
        Transaction(id=3, user_id="user1", name="name1", amount=2.99, date="2024-01-14"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-15"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-16"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-29"),
        Transaction(id=4, user_id="user1", name="name1", amount=2.99, date="2024-01-31"),
    ]
    assert get_pct_transactions_days_apart(transactions[0], transactions, 14, 0) == 2 / 7
    assert get_pct_transactions_days_apart(transactions[0], transactions, 14, 1) == 4 / 7


def test_get_is_insurance() -> None:
    """Test get_is_insurance."""
    assert get_is_insurance(
        Transaction(id=1, user_id="user1", name="Allstate Insurance", amount=100, date="2024-01-01")
    )
    assert not get_is_insurance(Transaction(id=2, user_id="user1", name="AT&T", amount=100, date="2024-01-01"))


def test_get_is_phone() -> None:
    """Test get_is_phone."""
    assert get_is_phone(Transaction(id=2, user_id="user1", name="AT&T", amount=100, date="2024-01-01"))
    assert not get_is_phone(Transaction(id=3, user_id="user1", name="Duke Energy", amount=200, date="2024-01-02"))


def test_get_is_utility() -> None:
    """Test get_is_utility."""
    assert get_is_utility(Transaction(id=3, user_id="user1", name="Duke Energy", amount=200, date="2024-01-02"))
    assert not get_is_utility(
        Transaction(id=4, user_id="user1", name="HighEnergy Soft Drinks", amount=2.99, date="2024-01-03")
    )


def test_get_is_always_recurring() -> None:
    """Test get_is_always_recurring."""
    assert get_is_always_recurring(Transaction(id=1, user_id="user1", name="netflix", amount=100, date="2024-01-01"))
    assert not get_is_always_recurring(
        Transaction(id=2, user_id="user1", name="walmart", amount=100, date="2024-01-01")
    )


def test_get_transaction_z_score() -> None:
    """Test get_transaction_z_score."""
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=100, date="2024-01-01"),
    ]
    assert get_transaction_z_score(transactions[0], transactions) == 0

    # Test with varying amounts
    transactions = [
        Transaction(id=1, user_id="user1", name="name1", amount=90, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="name1", amount=100, date="2024-01-01"),
        Transaction(id=3, user_id="user1", name="name1", amount=110, date="2024-01-01"),
    ]
    # Use approximate comparison with pytest
    z_score = get_transaction_z_score(transactions[0], transactions)
    assert -1.3 < z_score < -1.1  # Allow a small tolerance for floating-point precision


def test_get_is_amazon_prime() -> None:
    """Test get_is_amazon_prime."""
    assert get_is_amazon_prime(Transaction(id=1, user_id="user1", name="amazon prime", amount=100, date="2024-01-01"))
    assert not get_is_amazon_prime(Transaction(id=2, user_id="user1", name="netflix", amount=100, date="2024-01-01"))


def test_get_is_known_recurring_vendor() -> None:
    """Test get_is_known_recurring_vendor."""
    trans1 = Transaction("Netflix", 15.99, "01/01/2023")
    trans2 = Transaction("Walmart", 50.00, "01/01/2023")
    trans3 = Transaction("NETFLIX", 15.99, "01/01/2023")
    assert get_is_known_recurring_vendor(trans1) is True
    assert get_is_known_recurring_vendor(trans2) is False
    assert get_is_known_recurring_vendor(trans3) is True


def test_get_is_known_non_recurring_vendor() -> None:
    """Test get_is_known_non_recurring_vendor."""
    trans1 = Transaction("Walmart", 50.00, "01/01/2023")
    trans2 = Transaction("Netflix", 15.99, "01/01/2023")
    trans3 = Transaction("Star Bucks", 5.00, "01/01/2023")
    assert get_is_known_non_recurring_vendor(trans1) is True
    assert get_is_known_non_recurring_vendor(trans2) is False
    assert get_is_known_non_recurring_vendor(trans3) is True


def test_get_same_amount_count() -> None:
    """Test get_same_amount_count."""
    trans1 = [
        Transaction("Test", 10.00, "01/01/2023"),
        Transaction("Test", 10.00, "02/01/2023"),
        Transaction("Test", 10.00, "03/01/2023"),
    ]
    trans2 = [
        Transaction("Test", 10.00, "01/01/2023"),
        Transaction("Test", 10.09, "02/01/2023"),
        Transaction("Test", 9.91, "03/01/2023"),
    ]
    trans3 = [Transaction("Test", 10.00, "01/01/2023"), Transaction("Test", 11.00, "02/01/2023")]
    assert get_same_amount_count(trans1) == 3
    assert get_same_amount_count(trans2) == 3
    assert get_same_amount_count(trans3) == 1
    assert get_same_amount_count([]) == 0


def test_get_is_albert_99_recurring() -> None:
    """Test get_is_albert_99_recurring."""
    trans1 = Transaction("Albert Subscription", 1.99, "01/01/2023")
    trans2 = Transaction("Albert Subscription", 1.50, "01/01/2023")
    trans3 = Transaction("Netflix", 1.99, "01/01/2023")
    trans4 = Transaction("ALBERT SUB", 1.99, "01/01/2023")
    assert get_is_albert_99_recurring(trans1) is True
    assert get_is_albert_99_recurring(trans2) is False
    assert get_is_albert_99_recurring(trans3) is False
    assert get_is_albert_99_recurring(trans4) is True


def test_get_amount_consistency_score() -> None:
    """Test get_amount_consistency_score."""
    trans1 = [
        Transaction("Test", 10.00, "01/01/2023"),
        Transaction("Test", 10.00, "02/01/2023"),
        Transaction("Test", 10.00, "03/01/2023"),
    ]
    trans2 = [
        Transaction("Test", 10.00, "01/01/2023"),
        Transaction("Test", 10.40, "02/01/2023"),
        Transaction("Test", 10.30, "03/01/2023"),
    ]
    trans3 = [
        Transaction("Test", 50.00, "01/01/2023"),
        Transaction("Test", 55.00, "02/01/2023"),
        Transaction("Test", 60.00, "03/01/2023"),
    ]
    trans4 = [Transaction("Test", 0.00, "01/01/2023"), Transaction("Test", 0.00, "02/01/2023")]
    assert get_amount_consistency_score(trans1) == 1.0
    assert get_amount_consistency_score(trans2) == 1.0
    assert get_amount_consistency_score(trans3) == 0.0
    assert get_amount_consistency_score(trans4) == 0.0
    assert get_amount_consistency_score([]) == 0.0


def test_get_interval_consistency_score() -> None:
    """Test get_interval_consistency_score."""
    trans1 = [
        Transaction("Netflix", 15.99, "01/01/2023"),
        Transaction("Netflix", 15.99, "02/01/2023"),
        Transaction("Netflix", 15.99, "03/01/2023"),
    ]
    trans2 = [
        Transaction("Gym", 20.00, "01/01/2023"),
        Transaction("Gym", 20.00, "01/15/2023"),
        Transaction("Gym", 20.00, "01/29/2023"),
    ]
    trans3 = [Transaction("Membership", 100.00, "01/01/2023"), Transaction("Membership", 100.00, "01/01/2024")]
    trans4 = [
        Transaction("Test", 10.00, "01/01/2023"),
        Transaction("Test", 10.00, "01/15/2023"),
        Transaction("Test", 10.00, "02/15/2023"),
    ]
    assert pytest.approx(get_interval_consistency_score(trans1)) == 0.5
    assert get_interval_consistency_score(trans2) == 1.0
    assert get_interval_consistency_score(trans3) == 1.0
    assert get_interval_consistency_score(trans4) == 0.0
    assert get_interval_consistency_score([]) == 0.0


def test_get_combined_recurrence_score() -> None:
    """Test get_combined_recurrence_score."""
    trans1 = [
        Transaction("Netflix", 15.99, "01/01/2023"),
        Transaction("Netflix", 15.99, "02/01/2023"),
        Transaction("Netflix", 15.99, "03/01/2023"),
    ]
    trans2 = [
        Transaction("Walmart", 15.99, "01/01/2023"),
        Transaction("Walmart", 15.99, "02/01/2023"),
        Transaction("Walmart", 15.99, "03/01/2023"),
    ]
    trans3 = [
        Transaction("Netflix", 50.00, "01/01/2023"),
        Transaction("Netflix", 55.00, "02/01/2023"),
        Transaction("Netflix", 60.00, "03/01/2023"),
    ]
    assert pytest.approx(get_combined_recurrence_score(trans1[0], trans1)) == 0.85
    assert pytest.approx(get_combined_recurrence_score(trans2[0], trans2)) == 0.25
    assert pytest.approx(get_combined_recurrence_score(trans3[0], trans3)) == 0.45
    assert get_combined_recurrence_score(trans1[0], []) == 0.0


def test_get_is_recurring_same_amount_specific_intervals() -> None:
    """Test get_is_recurring_same_amount_specific_intervals."""
    trans1 = [
        Transaction("Netflix", 15.99, "01/01/2023"),
        Transaction("Netflix", 15.99, "02/01/2023"),
        Transaction("Netflix", 15.99, "03/01/2023"),
    ]
    trans2 = [
        Transaction("Gym", 20.00, "01/01/2023"),
        Transaction("Gym", 20.00, "01/15/2023"),
        Transaction("Gym", 20.00, "01/29/2023"),
    ]
    trans3 = [Transaction("Membership", 100.00, "01/01/2023"), Transaction("Membership", 100.00, "01/01/2024")]
    trans4 = [
        Transaction("Utility", 50.00, "01/01/2023"),
        Transaction("Utility", 55.00, "02/01/2023"),
        Transaction("Utility", 60.00, "03/01/2023"),
    ]
    assert get_is_recurring_same_amount_specific_intervals(trans1) is True
    assert get_is_recurring_same_amount_specific_intervals(trans2) is True
    assert get_is_recurring_same_amount_specific_intervals(trans3) is False
    assert get_is_recurring_same_amount_specific_intervals(trans4) is False
    assert get_is_recurring_same_amount_specific_intervals([]) is False


def test_detect_monthly_with_missing_entries() -> None:
    """Test detect_monthly_with_missing_entries."""
    trans1 = [
        Transaction("Netflix", 15.99, "01/01/2023"),
        Transaction("Netflix", 15.99, "02/01/2023"),
        Transaction("Netflix", 15.99, "03/01/2023"),
    ]
    trans2 = [Transaction("Test", 10.00, "01/01/2023"), Transaction("Test", 10.00, "03/01/2023")]
    trans3 = [
        Transaction("Utility", 50.00, "01/01/2023"),
        Transaction("Utility", 55.00, "02/01/2023"),
        Transaction("Utility", 60.00, "03/01/2023"),
    ]
    assert detect_monthly_with_missing_entries(trans1) == 1
    assert detect_monthly_with_missing_entries(trans2) == 0
    assert detect_monthly_with_missing_entries(trans3) == 0
    assert detect_monthly_with_missing_entries([]) == 0
