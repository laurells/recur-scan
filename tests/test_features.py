import pytest

from recur_scan.features import (
    average_transaction_amount,
    get_day,
    get_day_of_week,
    get_month,
    get_n_transactions_same_amount,
    get_percent_transactions_same_amount,
    identical_transaction_ratio,
    is_fixed_interval,
    is_large_transaction,
    is_monthly_recurring,
    is_price_increasing,
    is_start_or_end_of_month,
    median_transaction_interval,
    most_common_transaction_day,
    normalized_transaction_frequency,
    standard_deviation_of_transaction_amount,
    subscription_price_consistency,
    test_get_n_transactions_same_amount,
    test_get_percent_transactions_same_amount,
    time_since_last_transaction,
    time_since_last_transaction_same_amount,
    transaction_amount_ratio,
    transaction_frequency,
    transaction_time_fourier,
)
from recur_scan.transactions import Transaction


@pytest.fixture
def transactions():
    """Fixture providing test transactions."""
    return [
        Transaction(id=1, user_id="user1", name="vendor1", amount=100.0, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="vendor1", amount=100.0, date="2024-01-02"),
        Transaction(id=3, user_id="user1", name="vendor1", amount=200.0, date="2024-01-03"),
        Transaction(id=4, user_id="user2", name="vendor2", amount=200.0, date="2024-01-04"),
        Transaction(id=5, user_id="user1", name="vendor1", amount=100.0, date="2024-01-08"),
        Transaction(id=6, user_id="user1", name="vendor1", amount=100.0, date="2024-01-15"),
        Transaction(id=7, user_id="user1", name="vendor1", amount=100.0, date="2024-01-22"),
        Transaction(id=8, user_id="user1", name="vendor1", amount=200.0, date="2024-01-29"),
    ]


def test_get_n_transactions_same_amount(transactions) -> None:
    transaction = transactions[0]
    assert get_n_transactions_same_amount(transaction, transactions) == 3


def test_get_percent_transactions_same_amount(transactions) -> None:
    transaction = transactions[0]
    assert pytest.approx(get_percent_transactions_same_amount(transaction, transactions)) == 3 / 8


def test_get_day_of_week(transactions) -> None:
    transaction = transactions[0]
    assert get_day_of_week(transaction) == "Monday"


def test_get_month(transactions) -> None:
    transaction = transactions[0]
    assert get_month(transaction) == 1


def test_get_day(transactions) -> None:
    transaction = transactions[0]
    assert get_day(transaction) == 1


def test_time_since_last_transaction(transactions) -> None:
    transaction = transactions[2]
    assert time_since_last_transaction(transaction, transactions) == 1


def test_time_since_last_transaction_same_amount(transactions) -> None:
    transaction = transactions[2]
    assert time_since_last_transaction_same_amount(transaction, transactions) == 0


def test_average_transaction_amount(transactions) -> None:
    assert average_transaction_amount("user1", transactions) == 133.33


def test_transaction_frequency(transactions) -> None:
    assert transaction_frequency("user1", transactions, 30) == 5


def test_standard_deviation_of_transaction_amount(transactions) -> None:
    assert round(standard_deviation_of_transaction_amount("user1", transactions), 2) == 50.0


def test_most_common_transaction_day(transactions) -> None:
    assert most_common_transaction_day("user1", transactions) == 1


def test_is_fixed_interval(transactions) -> None:
    transaction = transactions[2]
    assert is_fixed_interval(transaction, transactions) == 1


def test_subscription_price_consistency(transactions) -> None:
    assert subscription_price_consistency("user1", transactions) == 0.0


def test_is_price_increasing(transactions) -> None:
    assert is_price_increasing("user1", transactions) == 1


def test_transaction_time_fourier(transactions) -> None:
    assert transaction_time_fourier("user1", transactions) >= 0


def test_identical_transaction_ratio(transactions) -> None:
    transaction = transactions[0]
    assert identical_transaction_ratio(transaction, transactions) == 0.25


def test_is_start_or_end_of_month(transactions) -> None:
    transaction = transactions[0]
    assert is_start_or_end_of_month(transaction) == 1


def test_is_monthly_recurring(transactions) -> None:
    transaction = transactions[0]
    assert is_monthly_recurring(transaction, transactions) == 1


def test_is_large_transaction(transactions) -> None:
    """Test that is_large_transaction returns 1 for large transactions."""
    transaction = Transaction(id=9, user_id="user1", name="vendor1", amount=400, date="2024-01-10")
    assert is_large_transaction(transaction, transactions) == 1

    transaction = Transaction(id=10, user_id="user1", name="vendor1", amount=200, date="2024-01-11")
    assert is_large_transaction(transaction, transactions) == 0


def test_transaction_amount_ratio(transactions) -> None:
    """Test that transaction_amount_ratio returns the correct ratio."""
    transaction = transactions[0]
    assert transaction_amount_ratio(transaction, transactions) == 0.75

    transaction = transactions[2]
    assert transaction_amount_ratio(transaction, transactions) == 1.5


def test_median_transaction_interval(transactions) -> None:
    """Test that median_transaction_interval returns the correct median interval."""
    assert median_transaction_interval("user1", transactions) == 7.0


def test_normalized_transaction_frequency(transactions) -> None:
    """Test that normalized_transaction_frequency returns the correct frequency."""
    assert normalized_transaction_frequency("user1", transactions) == 0.25
