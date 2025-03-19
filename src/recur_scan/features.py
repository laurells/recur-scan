from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Union, Optional

import numpy as np

from recur_scan.transactions import Transaction


# Helper Functions
def _parse_date(date_string: str) -> datetime | None:
    """Parse a date string into a datetime object with error handling.

    Args:
        date_string (str): Date in 'YYYY-MM-DD' format.

    Returns:
        Union[datetime, None]: Parsed datetime object or None if parsing fails.
    """
    try:
        parsed_date = datetime.strptime(date_string, "%Y-%m-%d")
        return parsed_date
    except ValueError:
        return None


def _aggregate_transactions(transactions: list[Transaction]) -> dict[str, dict[str, list[Transaction]]]:
    """Group transactions by user ID and merchant name for efficient feature computation.

    Args:
        transactions (List[Transaction]): List of all transaction objects.

    Returns:
        Dict[str, Dict[str, List[Transaction]]]: Nested dictionary with user_id as outer key,
        merchant_name as inner key, and list of transactions as values.
    """
    user_merchant_groups: dict[str, dict[str, list[Transaction]]] = {}
    for transaction in transactions:
        user_id = transaction.user_id
        merchant_name = transaction.name
        # Initialize dictionary for user if not present
        if user_id not in user_merchant_groups:
            user_merchant_groups[user_id] = {}
        # Initialize list for merchant if not present
        if merchant_name not in user_merchant_groups[user_id]:
            user_merchant_groups[user_id][merchant_name] = []
        # Append transaction to the appropriate group
        user_merchant_groups[user_id][merchant_name].append(transaction)
    return user_merchant_groups


def _calculate_intervals(dates: list[datetime]) -> list[int]:
    """Calculate the number of days between consecutive dates in a sorted list.

    Args:
        dates (List[datetime]): List of datetime objects, assumed to be sorted.

    Returns:
        List[int]: List of intervals in days between consecutive dates; empty if fewer than 2 dates.
    """
    intervals: list[int] = []
    # Need at least 2 dates to compute an interval
    if len(dates) < 2:
        return intervals
    # Compute days between each pair of consecutive dates
    for i in range(1, len(dates)):
        current_date = dates[i]
        previous_date = dates[i - 1]
        days_between = (current_date - previous_date).days
        intervals.append(days_between)
    return intervals


def _calculate_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean and standard deviation of a list of numbers.

    Args:
        values (List[float]): List of numerical values (e.g., intervals or amounts).

    Returns:
        Dict[str, float]: Dictionary with 'mean' and 'std' keys; both 0.0 if list is empty.
    """
    stats = {}
    # Handle empty list case
    if len(values) == 0:
        stats["mean"] = 0.0
        stats["std"] = 0.0
        return stats
    # Calculate mean manually
    total = 0.0
    for num in values:
        total += num
    mean_value = total / len(values)
    # Use NumPy for efficient standard deviation calculation
    std_value = float(np.std(values))
    stats["mean"] = mean_value
    stats["std"] = std_value
    return stats


# Individual Feature Functions
def n_transactions_same_amount_feature(transaction: Transaction, amount_counts: dict[float, int]) -> int:
    """Count how many transactions across all users have the same amount as the given transaction.

    Args:
        transaction (Transaction): The transaction to evaluate.
        all_transactions (List[Transaction]): List of all transactions (unused here but kept for consistency).
        amount_counts (Dict[float, int]): Precomputed counts of transactions per amount.

    Returns:
        int: Number of transactions with the same amount; 0 if not found in amount_counts.
    """
    return amount_counts.get(transaction.amount, 0)


def percent_transactions_same_amount_feature(
    transaction: Transaction, all_transactions: list[Transaction], amount_counts: dict[float, int]
) -> float:
    """Calculate the percentage of all transactions that match the given transaction's amount.

    Args:
        transaction (Transaction): The transaction to evaluate.
        all_transactions (List[Transaction]): List of all transactions to compute total count.
        amount_counts (Dict[float, int]): Precomputed counts of transactions per amount.

    Returns:
        float: Ratio of transactions with the same amount to total transactions; 0.0 if no transactions.
    """
    n_transactions_same_amount = amount_counts.get(transaction.amount, 0)
    return n_transactions_same_amount / len(all_transactions) if all_transactions else 0.0


def identical_transaction_ratio_feature(
    transaction: Transaction, all_transactions: list[Transaction], merchant_trans: list[Transaction]
) -> float:
    """Calculate the ratio of merchant-specific transactions with the same amount and name to all transactions.

    Args:
        transaction (Transaction): The transaction to evaluate.
        all_transactions (List[Transaction]): List of all transactions to compute total count.
        merchant_trans (List[Transaction]): List of transactions for this user and merchant.

    Returns:
        float: Ratio of identical transactions to total transactions; 0.0 if no transactions.
    """
    identical_transaction_count = 0
    for t in merchant_trans:
        if t.amount == transaction.amount and t.name == transaction.name:
            identical_transaction_count += 1
    return identical_transaction_count / len(all_transactions) if len(all_transactions) > 0 else 0.0


def is_monthly_recurring_feature(merchant_trans: list[Transaction]) -> int:
    """Determine if transactions occur on roughly the same day each month (≤3 unique days).

    Args:
        merchant_trans (List[Transaction]): List of transactions for this user and merchant.

    Returns:
        int: 1 if transactions occur on ≤3 unique days of the month, 0 otherwise.
    """
    unique_days = set()
    for trans in merchant_trans:
        date = _parse_date(trans.date)
        if date is not None:
            day_of_month = date.day
            unique_days.add(day_of_month)
    return 1 if len(unique_days) <= 5 else 0


def is_varying_amount_recurring_feature(interval_stats: dict[str, float], amount_stats: dict[str, float]) -> int:
    """Identify recurring transactions with varying amounts but consistent intervals.

    Args:
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.
        amount_stats (Dict[str, float]): Mean and std of transaction amounts.

    Returns:
        int: 1 if intervals are consistent (<45 days std) and amounts vary (>0.002 std/mean), 0 otherwise.
    """
    is_varying_amount_recurring = 0
    # Check if intervals are reasonably consistent and variation in amounts
    if interval_stats["std"] < 45 and amount_stats["mean"] > 0 and (amount_stats["std"] / amount_stats["mean"]) > 0.002:
        is_varying_amount_recurring = 1
    return is_varying_amount_recurring


def day_consistency_score_feature(merchant_trans: list[Transaction]) -> float:
    """Measure consistency of transaction days within a month (0 to 1 scale).

    Args:
        merchant_trans (List[Transaction]): List of transactions for this user and merchant.

    Returns:
        float: Score from 0 to 1; higher means more consistent days (lower std).
    """
    day_values = []
    for trans in merchant_trans:
        date = _parse_date(trans.date)
        if date is not None:
            day_values.append(date.day)
    day_consistency_score = 0.0
    if len(day_values) > 0:
        day_std = float(np.std(day_values))  # Standard deviation of days
        consistency_ratio = day_std / 5.0  # Normalize by 5 days
        if consistency_ratio > 1.0:
            consistency_ratio = 1.0
        day_consistency_score = 1.0 - consistency_ratio  # Invert: lower std → higher score
    return day_consistency_score


def is_near_periodic_interval_feature(interval_stats: dict[str, float]) -> int:
    """Check if transaction intervals are near weekly, monthly, or yearly periodicity.

    Args:
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.

    Returns:
        int: 1 if mean interval is close to 7, 30, or 365 days with low std (<7), 0 otherwise.
    """
    is_near_periodic_interval = 0
    mean_interval = interval_stats["mean"]
    periodic_targets = [(7, 3), (30, 5), (365, 15)]  # (target days, tolerance)
    for target, tolerance in periodic_targets:
        difference = abs(mean_interval - target)
        if difference <= tolerance and interval_stats["std"] < 7:  # Check proximity and consistency
            is_near_periodic_interval = 1
            break
    return is_near_periodic_interval


def merchant_amount_std_feature(amount_stats: dict[str, float]) -> float:
    """Calculate normalized standard deviation of transaction amounts for the merchant.

    Args:
        amount_stats (Dict[str, float]): Mean and std of transaction amounts.

    Returns:
        float: Std divided by mean; 0.0 if mean is 0 to avoid division by zero.
    """
    return amount_stats["std"] / amount_stats["mean"] if amount_stats["mean"] > 0 else 0.0


def merchant_interval_std_feature(interval_stats: dict[str, float]) -> float:
    """Extract standard deviation of intervals between transactions.

    Args:
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.

    Returns:
        float: Standard deviation of intervals; 0.0 if no intervals.
    """
    return interval_stats["std"]


def merchant_interval_mean_feature(interval_stats: dict[str, float]) -> float:
    """Extract mean of intervals between transactions.

    Args:
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.

    Returns:
        float: Mean interval in days; 0.0 if no intervals.
    """
    return interval_stats["mean"]


def time_since_last_transaction_same_merchant_feature(parsed_dates: list[datetime]) -> float:
    intervals = _calculate_intervals(parsed_dates)
    return sum(intervals) / len(intervals) if intervals else 0.0


def is_deposit_feature(transaction: Transaction, merchant_trans: list[Transaction]) -> int:
    """Identify if the transaction might be a recurring deposit based on amount and frequency.

    Args:
        transaction (Transaction): The transaction to evaluate.
        merchant_trans (List[Transaction]): List of transactions for this user and merchant.

    Returns:
        int: 1 if amount is positive and there are ≥3 transactions, 0 otherwise.
    """
    is_deposit = 0
    if transaction.amount > 0 and len(merchant_trans) >= 3:
        is_deposit = 1
    return is_deposit


def day_of_week_feature(transaction: Transaction) -> int:
    """Day of the week (0-6, Monday-Sunday)."""
    date = _parse_date(transaction.date)
    return date.weekday() if date else 0


def transaction_month_feature(transaction: Transaction) -> int:
    """Month of the transaction (1-12)."""
    date = _parse_date(transaction.date)
    return date.month if date else 0


def rolling_amount_mean_feature(merchant_trans: list[Transaction]) -> float:
    """Rolling mean of last 3 transaction amounts."""
    amounts = [t.amount for t in merchant_trans[-3:]]  # Last 3 transactions
    return sum(amounts) / len(amounts) if amounts else 0.0


def low_amount_variation_feature(amount_stats: dict[str, float]) -> int:
    """Indicator for low amount variation (std/mean ≤ 0.1)."""
    ratio = amount_stats["std"] / amount_stats["mean"] if amount_stats["mean"] > 0 else float("inf")
    return 1 if ratio <= 0.1 else 0


def recurrence_likelihood_feature(
    merchant_trans: list[Transaction], interval_stats: dict[str, float], amount_stats: dict[str, float]
) -> float:
    interval_std = interval_stats["std"]
    interval_score = 1 / (interval_std / 10.0 + 1)
    mean_amount = amount_stats["mean"]
    amount_score = 1 / (amount_stats["std"] / (mean_amount + 0.01) + 1)
    frequency_score = min(len(merchant_trans) / 5.0, 1.0)
    return interval_score * amount_score * frequency_score


# Main Feature Extraction Function
def get_features(
    transaction: Transaction,
    all_transactions: list[Transaction],
    groups: dict[str, dict[str, list[Transaction]]] | None = None,
    amount_counts: dict[float, int] | None = None,
) -> dict[str, float | int]:
    """Extract all features for a transaction by calling individual feature functions.
    This prepares a dictionary of features for model training.

    Args:
        transaction (Transaction): The transaction to extract features for.
        all_transactions (List[Transaction]): List of all transactions for context.
        groups (Dict[str, Dict[str, List[Transaction]]], optional): Precomputed user-merchant groups.
        amount_counts (Dict[float, int], optional): Precomputed counts of transactions per amount.

    Returns:
        Dict[str, Union[float, int]]: Dictionary mapping feature names to their computed values.
    """
    # Use precomputed groups if provided, otherwise compute them
    if groups is None:
        groups = _aggregate_transactions(all_transactions)
    # Use precomputed amount counts if provided, otherwise compute them
    if amount_counts is None:
        amount_counts = defaultdict(int)
        for t in all_transactions:
            amount_counts[t.amount] += 1

    # Extract user ID and merchant name from the transaction
    user_id, merchant_name = transaction.user_id, transaction.name
    # Get transactions for this user and merchant
    merchant_trans = groups.get(user_id, {}).get(merchant_name, [])
    # Sort transactions by date for chronological analysis
    merchant_trans.sort(key=lambda x: x.date)

    # Parse all dates for this merchant's transactions once
    parsed_dates = []
    for trans in merchant_trans:
        date = _parse_date(trans.date)
        if date is not None:
            parsed_dates.append(date)

    # Calculate intervals and amounts for statistical analysis
    intervals = _calculate_intervals(parsed_dates)
    amounts = [trans.amount for trans in merchant_trans]
    interval_stats = _calculate_statistics([float(i) for i in intervals])
    amount_stats = _calculate_statistics(amounts)

    # Construct feature dictionary by calling each feature function
    features_dict = {
        "n_transactions_same_amount": n_transactions_same_amount_feature(transaction, amount_counts),
        "percent_transactions_same_amount": percent_transactions_same_amount_feature(
            transaction, all_transactions, amount_counts
        ),
        "identical_transaction_ratio": identical_transaction_ratio_feature(
            transaction, all_transactions, merchant_trans
        ),
        "is_monthly_recurring": is_monthly_recurring_feature(merchant_trans),
        "recurrence_likelihood": recurrence_likelihood_feature(merchant_trans, interval_stats, amount_stats),
        "is_varying_amount_recurring": is_varying_amount_recurring_feature(interval_stats, amount_stats),
        "day_consistency_score": day_consistency_score_feature(merchant_trans),
        "is_near_periodic_interval": is_near_periodic_interval_feature(interval_stats),
        "merchant_amount_std": merchant_amount_std_feature(amount_stats),
        "merchant_interval_std": merchant_interval_std_feature(interval_stats),
        "merchant_interval_mean": merchant_interval_mean_feature(interval_stats),
        "time_since_last_transaction_same_merchant": time_since_last_transaction_same_merchant_feature(parsed_dates),
        "is_deposit": is_deposit_feature(transaction, merchant_trans),
        "day_of_week": day_of_week_feature(transaction),
        "transaction_month": transaction_month_feature(transaction),
        "rolling_amount_mean": rolling_amount_mean_feature(merchant_trans),
        "low_amount_variation": low_amount_variation_feature(amount_stats),
    }

    return features_dict
