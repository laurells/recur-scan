import re
from collections import defaultdict
from datetime import datetime
from functools import lru_cache

import numpy as np
from scipy.stats import entropy

from recur_scan.transactions import Transaction


# Helper Functions
@lru_cache(maxsize=1024)
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
# def n_transactions_same_amount(transaction: Transaction, amount_counts: dict[float, int]) -> int:
#     """Count how many transactions across all users have the same amount as the given transaction.

#     Args:
#         transaction (Transaction): The transaction to evaluate.
#         all_transactions (List[Transaction]): List of all transactions (unused here but kept for consistency).
#         amount_counts (Dict[float, int]): Precomputed counts of transactions per amount.

#     Returns:
#         int: Number of transactions with the same amount; 0 if not found in amount_counts.
#     """
#     return amount_counts.get(transaction.amount, 0)


# def percent_transactions_same_amount(
#     transaction: Transaction, all_transactions: list[Transaction], amount_counts: dict[float, int]
# ) -> float:
#     """Calculate the percentage of all transactions that match the given transaction's amount.

#     Args:
#         transaction (Transaction): The transaction to evaluate.
#         all_transactions (List[Transaction]): List of all transactions to compute total count.
#         amount_counts (Dict[float, int]): Precomputed counts of transactions per amount.

#     Returns:
#         float: Ratio of transactions with the same amount to total transactions; 0.0 if no transactions.
#     """
#     n_transactions_same_amount = amount_counts.get(transaction.amount, 0)
#     return n_transactions_same_amount / len(all_transactions) if all_transactions else 0.0


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


def is_monthly_recurring_feature(merchant_trans: list[Transaction]) -> float:
    """Determine if transactions occur on roughly the same day each month (≤3 unique days)."""
    if len(merchant_trans) <= 1:
        return 0.0
    days = [date.day for date in [_parse_date(t.date) for t in merchant_trans] if date]
    if not days:
        return 0.0
    unique_days = len(set(days))
    return 1.0 - min((unique_days - 1) / 5.0, 1.0)


def is_varying_amount_recurring_feature(interval_stats: dict[str, float], amount_stats: dict[str, float]) -> int:
    """Identify recurring transactions with varying amounts but consistent intervals.

    Args:
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.
        amount_stats (Dict[str, float]): Mean and std of transaction amounts.

    Returns:
        int: 1 if intervals are consistent (<45 days std) and amounts vary (>0.002 std/mean), 0 otherwise.
    """
    if interval_stats["std"] < 45 and amount_stats["mean"] > 0 and (amount_stats["std"] / amount_stats["mean"]) > 0.002:
        return 1
    return 0


def day_consistency_score_feature(merchant_trans: list[Transaction]) -> float:
    """Measure consistency of transaction days within a month (0 to 1 scale).

    Args:
        merchant_trans (List[Transaction]): List of transactions for this user and merchant.

    Returns:
        float: Score from 0 to 1; higher means more consistent days (lower std).
    """
    days = [date.day for date in [_parse_date(t.date) for t in merchant_trans] if date]
    if not days:
        return 0.0
    if len(days) == 1:
        return 0.5
    std = float(np.std(days))
    return 1.0 - min(std / 3.0, 1.0)


def is_near_periodic_interval_feature(interval_stats: dict[str, float]) -> float:
    """Score proximity to periodic intervals (0-1)."""
    if interval_stats["mean"] == 0:
        return 0.0
    mean = interval_stats["mean"]
    std = interval_stats["std"]
    targets = [(7, 2), (30, 3), (365, 10)]
    best_score = 0.0
    for target, tolerance in targets:
        deviation = abs(mean - target) / target
        if std < 5:  # Stricter consistency
            score = 1.0 - min(deviation / (tolerance / target), 1.0)
            best_score = max(best_score, score)
    return best_score


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
    return interval_stats["std"] if interval_stats["std"] > 0 else 30.0


def merchant_interval_mean_feature(interval_stats: dict[str, float]) -> float:
    """Extract mean of intervals between transactions.

    Args:
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.

    Returns:
        float: Mean interval in days; 0.0 if no intervals.
    """
    return interval_stats["mean"] if interval_stats["mean"] > 0 else 60.0


def time_since_last_transaction_same_merchant_feature(parsed_dates: list[datetime]) -> float:
    """Days since earliest transaction or average interval."""
    if not parsed_dates:
        return 0
    now = datetime.now()
    if len(parsed_dates) == 1:
        return (now - parsed_dates[0]).days / 365  # Normalize by year
    intervals = _calculate_intervals(parsed_dates)
    return (sum(intervals) / len(intervals)) / 365 if intervals else (now - min(parsed_dates)).days / 365


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


def day_of_week_feature(transaction: Transaction) -> float:
    """Day of the week (0-6, Monday-Sunday)."""
    date = _parse_date(transaction.date)
    return date.weekday() / 6 if date else 0


def transaction_month_feature(transaction: Transaction) -> float:
    """Month of the transaction (1-12)."""
    date = _parse_date(transaction.date)
    return (date.month - 1) / 11 if date else 0


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


def is_single_transaction_feature(merchant_trans: list[Transaction]) -> int:
    """Check if there is only one transaction for this user-merchant pair.

    Args:
        merchant_trans (List[Transaction]): List of transactions for the user and merchant.

    Returns:
        int: 1 if there is only one transaction, 0 otherwise.
    """
    return 1 if len(merchant_trans) == 1 else 0


def interval_variability_feature(interval_stats: dict[str, float]) -> float:
    """Calculate the coefficient of variation for transaction intervals.

    Args:
        interval_stats (Dict[str, float]): Dictionary with 'mean' and 'std' of intervals.

    Returns:
        float: Coefficient of variation (std / mean), or 0 if mean is 0.
    """
    mean = interval_stats["mean"]
    if mean == 0:
        return 1.0  # High variability for singletons
    return min(interval_stats["std"] / mean, 2.0)  # Cap at 2 for extreme cases


def merchant_amount_frequency_feature(merchant_trans: list[Transaction]) -> int:
    """Count the number of unique transaction amounts for the user-merchant pair.

    Args:
        merchant_trans (List[Transaction]): List of transactions for the user and merchant.

    Returns:
        int: Number of unique transaction amounts.
    """
    unique_amounts = {trans.amount for trans in merchant_trans}
    return len(unique_amounts)


def non_recurring_irregularity_score(
    merchant_trans: list[Transaction], interval_stats: dict[str, float], amount_stats: dict[str, float]
) -> float:
    """Calculate a score indicating likelihood of a transaction being non-recurring based on irregularity.

    Args:
        merchant_trans (List[Transaction]): List of transactions for the user and merchant.
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.
        amount_stats (Dict[str, float]): Mean and std of transaction amounts.

    Returns:
        float: Score between 0 and 1; higher values indicate more irregularity (non-recurring).
    """
    # Component 1: Interval variability (std/mean, capped at 1.0)
    mean_interval = interval_stats["mean"]
    interval_var = interval_stats["std"] / mean_interval if mean_interval > 0 else 0.0
    interval_score = min(interval_var, 1.0)

    # Component 2: Amount variability (std/mean from amount_stats, higher → more non-recurring)
    amount_var = amount_stats["std"] / amount_stats["mean"] if amount_stats["mean"] > 0 else 0.0
    amount_score = min(amount_var, 1.0)  # Cap at 1.0, high variability = non-recurring

    # Component 3: Inverse frequency (lower frequency → more non-recurring)
    frequency_score = 1.0 - min(len(merchant_trans) / 5.0, 1.0)

    # Weighted average
    weights = [0.4, 0.3, 0.3]  # Interval, amount, frequency
    total_score = weights[0] * interval_score + weights[1] * amount_score + weights[2] * frequency_score
    return min(total_score, 1.0)  # Ensure score stays in 0-1 range


def transaction_pattern_complexity(merchant_trans: list[Transaction], interval_stats: dict[str, float]) -> float:
    """Calculate a complexity score for the transaction pattern, higher for non-recurring.

    Args:
        merchant_trans (List[Transaction]): List of transactions for the user and merchant.
        interval_stats (Dict[str, float]): Mean and std of intervals between transactions.

    Returns:
        float: Score between 0 and 1; higher indicates more complex/non-recurring patterns.
    """
    parsed_dates = [date for date in [_parse_date(t.date) for t in merchant_trans] if date is not None]
    intervals = _calculate_intervals(parsed_dates)
    if not intervals:
        return 0.0

    # Component 1: Interval entropy (distribution complexity)
    bins = [min(max(1, int(i / 7)), 52) for i in intervals]
    value_counts = np.bincount(bins, minlength=53)[1:]
    interval_entropy = float(entropy(value_counts / value_counts.sum()) / np.log(52) if value_counts.sum() > 0 else 0.0)

    # Component 2: Interval variability (std/mean from interval_stats)
    mean_interval = interval_stats["mean"]
    interval_var = float(interval_stats["std"] / mean_interval if mean_interval > 0 else 0.0)
    interval_var_score = min(interval_var, 1.0)  # Cap at 1.0 for normalization

    # Component 3: Amount diversity
    amounts = [t.amount for t in merchant_trans]
    amount_entropy = float(min(len(set(amounts)) / 10.0, 1.0) if amounts else 0.0)

    # Component 4: Inverse frequency
    frequency_score = float(1.0 - min(len(merchant_trans) / 5.0, 1.0))

    # Weighted average
    weights = [0.4, 0.3, 0.2, 0.1]  # Entropy, variability, amount, frequency
    total_score = (
        weights[0] * interval_entropy
        + weights[1] * interval_var_score
        + weights[2] * amount_entropy
        + weights[3] * frequency_score
    )
    return float(min(total_score, 1.0))  # Ensure score stays in 0-1 range


def date_irregularity_dominance(
    merchant_trans: list[Transaction], interval_stats: dict[str, float], amount_stats: dict[str, float]
) -> float:
    """Score emphasizing date irregularity for non-recurring transactions.

    Args:
        merchant_trans (List[Transaction]): Transactions for the user and merchant.
        interval_stats (Dict[str, float]): Mean and std of intervals.
        amount_stats (Dict[str, float]): Mean and std of amounts.

    Returns:
        float: Score 0-1; higher indicates stronger non-recurring due to irregular dates.
    """
    # Explicitly filter None values for mypy
    parsed_dates = [date for date in [_parse_date(t.date) for t in merchant_trans] if date is not None]
    intervals = _calculate_intervals(parsed_dates)
    n_trans = len(merchant_trans)
    if n_trans <= 1:
        return 0.5

    bins = [min(max(1, int(i / 7)), 52) for i in intervals]
    value_counts = np.bincount(bins, minlength=53)[1:]
    entropy_score = float(entropy(value_counts / value_counts.sum()) / np.log(52) if value_counts.sum() > 0 else 0.0)
    mean = interval_stats["mean"]
    deviation = min(abs(mean - target) / target for target in [7, 30, 365])
    deviation_score = float(min(deviation * 3, 1.0))
    frequency_score = float(1.0 - min(n_trans / 5.0, 1.0))
    amount_consistency = float(
        1.0 - min(amount_stats["std"] / amount_stats["mean"] if amount_stats["mean"] > 0 else 0.0, 1.0)
    )
    weights = [0.5, 0.3, 0.1, 0.1]
    return float(
        min(
            weights[0] * entropy_score
            + weights[1] * deviation_score
            + weights[2] * frequency_score
            + weights[3] * amount_consistency,
            1.0,
        )
    )


def get_is_always_recurring(transaction: Transaction) -> bool:
    """Check if the transaction is always recurring because of the vendor name - check lowercase match"""
    always_recurring_vendors = {
        "google storage",
        "netflix",
        "hulu",
        "spotify",
    }
    return transaction.name.lower() in always_recurring_vendors


def get_is_insurance(transaction: Transaction) -> bool:
    """Check if the transaction is an insurance payment."""
    # use a regular expression with boundaries to match case-insensitive insurance
    # and insurance-related terms
    match = re.search(r"\b(insurance|insur|insuranc)\b", transaction.name, re.IGNORECASE)
    return bool(match)


def get_is_utility(transaction: Transaction) -> bool:
    """Check if the transaction is a utility payment."""
    # use a regular expression with boundaries to match case-insensitive utility
    # and utility-related terms
    match = re.search(r"\b(utility|utilit|energy)\b", transaction.name, re.IGNORECASE)
    return bool(match)


def get_is_phone(transaction: Transaction) -> bool:
    """Check if the transaction is a phone payment."""
    # use a regular expression with boundaries to match case-insensitive phone
    # and phone-related terms
    match = re.search(r"\b(at&t|t-mobile|verizon)\b", transaction.name, re.IGNORECASE)
    return bool(match)


def get_n_transactions_days_apart(
    transaction: Transaction,
    all_transactions: list[Transaction],
    n_days_apart: int,
    n_days_off: int,
) -> int:
    """
    Get the number of transactions in all_transactions that are within n_days_off of
    being n_days_apart from transaction.

    Args:
        transaction (Transaction): The reference transaction.
        all_transactions (List[Transaction]): List of transactions to compare against.
        n_days_apart (int): Target interval in days (e.g., 7 for weekly).
        n_days_off (int): Allowed deviation from the target interval.

    Returns:
        int: Number of transactions matching the criteria.
    """
    n_txs = 0
    transaction_date = _parse_date(transaction.date)
    if transaction_date is None:  # Early exit if reference date is invalid
        return 0

    # Pre-calculate bounds for faster checking
    lower_remainder = n_days_apart - n_days_off
    upper_remainder = n_days_off

    for t in all_transactions:
        t_date = _parse_date(t.date)
        if t_date is None:  # Skip if comparison date is invalid
            continue
        days_diff = abs((t_date - transaction_date).days)

        # Skip if the difference is less than minimum required
        if days_diff < n_days_apart - n_days_off:
            continue

        # Check if the difference is close to any multiple of n_days_apart
        remainder = days_diff % n_days_apart

        if remainder <= upper_remainder or remainder >= lower_remainder:
            n_txs += 1

    return n_txs


def get_pct_transactions_days_apart(
    transaction: Transaction, all_transactions: list[Transaction], n_days_apart: int, n_days_off: int
) -> float:
    """
    Get the percentage of transactions in all_transactions that are within
    n_days_off of being n_days_apart from transaction
    """
    return get_n_transactions_days_apart(transaction, all_transactions, n_days_apart, n_days_off) / len(
        all_transactions
    )


def _get_day(date: str) -> int:
    """Get the day of the month from a transaction date."""
    return int(date.split("-")[2])


def get_n_transactions_same_day(transaction: Transaction, all_transactions: list[Transaction], n_days_off: int) -> int:
    """Get the number of transactions in all_transactions that are on the same day of the month as transaction"""
    return len([t for t in all_transactions if abs(_get_day(t.date) - _get_day(transaction.date)) <= n_days_off])


def get_pct_transactions_same_day(
    transaction: Transaction, all_transactions: list[Transaction], n_days_off: int
) -> float:
    """Get the percentage of transactions in all_transactions that are on the same day of the month as transaction"""
    return get_n_transactions_same_day(transaction, all_transactions, n_days_off) / len(all_transactions)


def get_ends_in_99(transaction: Transaction) -> bool:
    """Check if the transaction amount ends in 99"""
    return (transaction.amount * 100) % 100 == 99


def get_n_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> int:
    """Get the number of transactions in all_transactions with the same amount as transaction"""
    return len([t for t in all_transactions if t.amount == transaction.amount])


def get_percent_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Get the percentage of transactions in all_transactions with the same amount as transaction"""
    if not all_transactions:
        return 0.0
    n_same_amount = len([t for t in all_transactions if t.amount == transaction.amount])
    return n_same_amount / len(all_transactions)


# Main Feature Extraction Function
def get_features(
    transaction: Transaction,
    all_transactions: list[Transaction],
) -> dict[str, float | int]:
    """Extract all features for a transaction by calling individual feature functions.
    This prepares a dictionary of features for model training.

    Args:
        transaction (Transaction): The transaction to extract features for.
        all_transactions (List[Transaction]): List of all transactions for context.

    Returns:
        Dict[str, Union[float, int]]: Dictionary mapping feature names to their computed values.
    """
    # Compute groups and amount counts internally
    groups = _aggregate_transactions(all_transactions)
    amount_counts: defaultdict[float, int] = defaultdict(int)
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
        # "n_transactions_same_amount": n_transactions_same_amount_feature(transaction, amount_counts),
        # "percent_transactions_same_amount": percent_transactions_same_amount_feature(
        #     transaction, all_transactions, amount_counts
        # ),
        "n_transactions_same_amount": get_n_transactions_same_amount(transaction, all_transactions),
        "percent_transactions_same_amount": get_percent_transactions_same_amount(transaction, all_transactions),
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
        "is_single_transaction": is_single_transaction_feature(merchant_trans),
        "interval_variability": interval_variability_feature(interval_stats),
        "merchant_amount_frequency": merchant_amount_frequency_feature(merchant_trans),
        "non_recurring_irregularity_score": non_recurring_irregularity_score(
            merchant_trans, interval_stats, amount_stats
        ),
        "transaction_pattern_complexity": transaction_pattern_complexity(merchant_trans, interval_stats),
        "date_irregularity_dominance": date_irregularity_dominance(merchant_trans, interval_stats, amount_stats),
        "ends_in_99": get_ends_in_99(transaction),
        "amount": transaction.amount,
        "same_day_exact": get_n_transactions_same_day(transaction, all_transactions, 0),
        "pct_transactions_same_day": get_pct_transactions_same_day(transaction, all_transactions, 0),
        "same_day_off_by_1": get_n_transactions_same_day(transaction, all_transactions, 1),
        "same_day_off_by_2": get_n_transactions_same_day(transaction, all_transactions, 2),
        "14_days_apart_exact": get_n_transactions_days_apart(transaction, all_transactions, 14, 0),
        "pct_14_days_apart_exact": get_pct_transactions_days_apart(transaction, all_transactions, 14, 0),
        "14_days_apart_off_by_1": get_n_transactions_days_apart(transaction, all_transactions, 14, 1),
        "pct_14_days_apart_off_by_1": get_pct_transactions_days_apart(transaction, all_transactions, 14, 1),
        "7_days_apart_exact": get_n_transactions_days_apart(transaction, all_transactions, 7, 0),
        "pct_7_days_apart_exact": get_pct_transactions_days_apart(transaction, all_transactions, 7, 0),
        "7_days_apart_off_by_1": get_n_transactions_days_apart(transaction, all_transactions, 7, 1),
        "pct_7_days_apart_off_by_1": get_pct_transactions_days_apart(transaction, all_transactions, 7, 1),
        "is_insurance": get_is_insurance(transaction),
        "is_utility": get_is_utility(transaction),
        "is_phone": get_is_phone(transaction),
        "is_always_recurring": get_is_always_recurring(transaction),
    }

    return features_dict
