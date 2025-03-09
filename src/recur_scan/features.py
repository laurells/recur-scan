import statistics
from datetime import datetime, timedelta

import numpy as np
from scipy.fft import fft

from recur_scan.transactions import Transaction


def get_n_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> int:
    """Get the number of transactions in all_transactions with the same amount as transaction"""
    return len([t for t in all_transactions if t.amount == transaction.amount])


def get_percent_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Get the percentage of transactions in all_transactions with the same amount as transaction"""
    if not all_transactions:
        return 0.0
    n_same_amount = len([t for t in all_transactions if t.amount == transaction.amount])
    return n_same_amount / len(all_transactions)


# New features
def get_day_of_week(transaction: Transaction) -> int:
    """Get the day of the week for the transaction date"""
    try:
        return datetime.strptime(transaction.date, "%Y-%m-%d").weekday()
    except ValueError:
        return -1


def get_month(transaction: Transaction) -> int:
    """Get the month for the transaction date"""
    try:
        return datetime.strptime(transaction.date, "%Y-%m-%d").month
    except ValueError:
        # Handle invalid date format
        return -1


def get_day(transaction: Transaction) -> int:
    """Get the day for the transaction date"""
    try:
        return datetime.strptime(transaction.date, "%Y-%m-%d").day
    except ValueError:
        # Handle invalid date format
        return -1


def time_since_last_transaction(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Calculate time since last transaction for the same user"""
    # Initialize a list to store transactions for the same user
    user_transactions = []

    # Collect transactions for the specified user
    for t in all_transactions:
        if t.user_id == transaction.user_id:
            user_transactions.append(t)

    # Sort the user's transactions by date
    user_transactions.sort(key=lambda x: x.date)

    # Check if there are at least two transactions to calculate the difference
    if len(user_transactions) < 2:
        return 0.0  # Not enough transactions to calculate time since last transaction

    # Get the date of the last transaction
    last_transaction_date = datetime.strptime(user_transactions[-2].date, "%Y-%m-%d")

    # Get the date of the current transaction
    current_transaction_date = datetime.strptime(transaction.date, "%Y-%m-%d")

    # Calculate the difference in days between the current transaction and the last transaction
    return (current_transaction_date - last_transaction_date).days


def time_since_last_transaction_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Calculate time since last transaction with the same amount for the same user"""
    # Initialize a list to store transactions for the same user and amount
    user_transactions = []

    # Collect transactions for the specified user and amount
    for t in all_transactions:
        if t.user_id == transaction.user_id and t.amount == transaction.amount:
            user_transactions.append(t)

    # Sort the user's transactions by date
    user_transactions.sort(key=lambda x: x.date)

    # Check if there are at least two transactions to calculate the difference
    if len(user_transactions) < 2:
        return 0.0  # Not enough transactions to calculate time since last transaction with the same amount

    # Get the date of the last transaction with the same amount
    last_transaction_date = datetime.strptime(user_transactions[-2].date, "%Y-%m-%d")

    # Get the date of the current transaction
    current_transaction_date = datetime.strptime(transaction.date, "%Y-%m-%d")

    # Calculate the difference in days between the current transaction and the last transaction with the same amount
    return (current_transaction_date - last_transaction_date).days


def average_transaction_amount(user_id: str, all_transactions: list[Transaction]) -> float:
    """Calculate the average transaction amount for the user"""
    # Initialize an empty list to store the user's transaction amounts
    user_transactions = []

    # Iterate through all transactions and collect amounts for the specified user
    for transaction in all_transactions:
        if transaction.user_id == user_id:
            user_transactions.append(transaction.amount)

    # Calculate the average transaction amount
    return float(np.mean(user_transactions)) if user_transactions else 0.0


def transaction_frequency(user_id: str, all_transactions: list[Transaction], period_days: int) -> int:
    """Count how many transactions a user has made in the last specified days"""
    # Calculate the cutoff date for the specified period
    cutoff_date = datetime.now() - timedelta(days=period_days)

    # Initialize a count for the number of transactions
    transaction_count = 0

    # Iterate through all transactions to count those within the specified period
    for transaction in all_transactions:
        if transaction.user_id == user_id:
            transaction_date = datetime.strptime(transaction.date, "%Y-%m-%d")
            if transaction_date >= cutoff_date:
                transaction_count += 1

    return transaction_count


def standard_deviation_of_transaction_amount(user_id: str, all_transactions: list[Transaction]) -> float:
    """Calculate the standard deviation of transaction amounts for the user"""
    # Initialize an empty list to store the user's transaction amounts
    user_transactions = []

    # Iterate through all transactions and collect amounts for the specified user
    for transaction in all_transactions:
        if transaction.user_id == user_id:
            user_transactions.append(transaction.amount)

    # Calculate the standard deviation of the transaction amounts
    return float(np.std(user_transactions, ddof=0)) if len(user_transactions) > 1 else 0.0


def change_in_average_transaction_amount(user_id: str, all_transactions: list[Transaction], period_days: int) -> float:
    """Calculate the change in average transaction amount over the specified period"""
    # Calculate the cutoff date for the specified period
    cutoff_date = datetime.now() - timedelta(days=period_days)

    # Initialize lists to store recent and overall transaction amounts
    recent_transactions = []
    overall_transactions = []

    # Iterate through all transactions to collect amounts for the specified user
    for transaction in all_transactions:
        if transaction.user_id == user_id:
            # Add to recent transactions if within the cutoff date
            transaction_date = datetime.strptime(transaction.date, "%Y-%m-%d")
            if transaction_date >= cutoff_date:
                recent_transactions.append(transaction.amount)
            # Always add to overall transactions
            overall_transactions.append(transaction.amount)

    # Calculate the average for recent and overall transactions
    recent_average = float(np.mean(recent_transactions)) if recent_transactions else 0.0
    overall_average = float(np.mean(overall_transactions)) if overall_transactions else 0.0

    # Return the change in average transaction amount
    return recent_average - overall_average


def change_in_transaction_frequency(user_id: str, all_transactions: list[Transaction], period_days: int) -> int:
    """Calculate the change in transaction frequency over the specified period"""
    # Calculate the cutoff date for the specified period
    cutoff_date = datetime.now() - timedelta(days=period_days)

    # Initialize counters for recent and overall transaction counts
    recent_count = 0
    overall_count = 0

    # Iterate through all transactions to count those for the specified user
    for transaction in all_transactions:
        if transaction.user_id == user_id:
            overall_count += 1  # Count all transactions for the user
            transaction_date = datetime.strptime(transaction.date, "%Y-%m-%d")
            if transaction_date >= cutoff_date:
                recent_count += 1  # Count recent transactions within the specified period

    # Return the change in transaction frequency
    return recent_count - overall_count


def transaction_interval_variance(user_id: str, all_transactions: list[Transaction]) -> float:
    """Measures the variance in days between consecutive transactions"""
    # Initialize a list to store the user's transactions
    user_transactions = []

    # Collect transactions for the specified user and sort them by date
    for transaction in all_transactions:
        if transaction.user_id == user_id:
            user_transactions.append(transaction)

    # Sort the user's transactions by date
    user_transactions.sort(key=lambda x: x.date)

    # Initialize a list to store intervals between transactions
    intervals = []

    # Calculate the intervals between consecutive transactions
    for i in range(1, len(user_transactions)):
        current_date = datetime.strptime(user_transactions[i].date, "%Y-%m-%d")
        previous_date = datetime.strptime(user_transactions[i - 1].date, "%Y-%m-%d")
        interval_days = (current_date - previous_date).days
        intervals.append(interval_days)

    # Calculate and return the variance of the intervals
    return float(np.var(intervals)) if len(intervals) > 1 else 0.0


def identical_transaction_ratio(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Finds the percentage of transactions with the same amount and merchant."""
    # Initialize a count for similar transactions
    similar_transaction_count = 0

    # Iterate through all transactions to count those that are identical
    for t in all_transactions:
        if t.amount == transaction.amount and t.name == transaction.name:
            similar_transaction_count += 1

    # Calculate the total number of transactions
    total_number_transactions = len(all_transactions)

    # Calculate and return the ratio of similar transactions to total transactions
    if total_number_transactions > 0:
        return similar_transaction_count / total_number_transactions
    else:
        return 0.0  # Return 0.0 if there are no transactions


def is_start_or_end_of_month(transaction: Transaction) -> int:
    """Returns 1 if the transaction occurs at the start (1st-3rd) or end (last 3 days) of the month."""
    try:
        # Parse the transaction date
        dt = datetime.strptime(transaction.date, "%Y-%m-%d")

        # Calculate the last day of the month
        last_day_of_month = (dt.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)

        # Check if the transaction date is at the start or end of the month
        if dt.day <= 3 or dt.day > last_day_of_month.day - 3:
            return 1  # Transaction is at the start or end of the month
        else:
            return 0  # Transaction is not at the start or end of the month
    except ValueError:
        return -1  # Invalid date format


def is_monthly_recurring(transaction: Transaction, all_transactions: list[Transaction]) -> int:
    """Returns 1 if the transaction happens on a similar day each month, otherwise 0"""
    # Initialize a set to store unique transaction days for the specified user and amount
    user_transaction_days = set()

    # Iterate through all transactions to collect days for the specified user and amount
    for t in all_transactions:
        if t.user_id == transaction.user_id and t.amount == transaction.amount:
            transaction_day = datetime.strptime(t.date, "%Y-%m-%d").day
            user_transaction_days.add(transaction_day)

    # Check if the number of unique transaction days is 3 or fewer
    return 1 if len(user_transaction_days) <= 3 else 0


def most_common_transaction_day(user_id: str, all_transactions: list[Transaction]) -> int:
    """Finds the most common day of the month the user transacts"""
    # Initialize a list to store the days of transactions for the specified user
    transaction_days = []

    # Iterate through all transactions to collect days for the specified user
    for t in all_transactions:
        if t.user_id == user_id:
            transaction_day = datetime.strptime(t.date, "%Y-%m-%d").day
            transaction_days.append(transaction_day)

    # Check if there are any transaction days recorded
    if not transaction_days:
        return -1  # Return -1 if there are no transactions

    # Find and return the most common transaction day
    return max(set(transaction_days), key=transaction_days.count)


def is_fixed_interval(transaction: Transaction, all_transactions: list[Transaction]) -> int:
    """Returns 1 if the transaction follows a strict interval (7, 14, or 30 days)"""
    # Initialize a list to store the user's transactions
    user_transactions = []

    # Collect transactions for the specified user and amount
    for t in all_transactions:
        if t.user_id == transaction.user_id and t.amount == transaction.amount:
            user_transactions.append(t)

    # Sort the user's transactions by date
    user_transactions.sort(key=lambda x: x.date)

    # Check if there are at least two transactions to calculate intervals
    if len(user_transactions) < 2:
        return 0  # Not enough transactions to determine interval

    # Initialize a list to store intervals between transactions
    intervals = []

    # Calculate the intervals between consecutive transactions
    for i in range(1, len(user_transactions)):
        current_date = datetime.strptime(user_transactions[i].date, "%Y-%m-%d")
        previous_date = datetime.strptime(user_transactions[i - 1].date, "%Y-%m-%d")
        interval_days = (current_date - previous_date).days
        intervals.append(interval_days)

    # Check if the standard deviation of intervals is less than 3
    return 1 if np.std(intervals) < 3 else 0


def subscription_price_consistency(user_id: str, all_transactions: list[Transaction]) -> float:
    """Calculates how much the transaction amounts deviate from the mean (normalized)"""
    # Initialize a list to store the user's transaction amounts
    user_transactions = []

    # Collect transactions for the specified user
    for t in all_transactions:
        if t.user_id == user_id:
            user_transactions.append(t.amount)

    # Check if there are enough transactions to calculate consistency
    if len(user_transactions) < 2:
        return 1.0  # Default to high consistency

    # Calculate the mean of the transaction amounts
    mean = statistics.mean(user_transactions)

    # Calculate and return the normalized standard deviation
    return statistics.stdev(user_transactions) / mean if mean > 0 else 0.0


def is_price_increasing(user_id: str, all_transactions: list[Transaction]) -> int:
    """Returns 1 if the user's transaction amounts are increasing over time"""
    # Initialize a list to store the user's transaction amounts
    user_transactions = []

    # Collect transactions for the specified user
    for t in all_transactions:
        if t.user_id == user_id:
            user_transactions.append(t.amount)

    # Sort the user's transaction amounts in ascending order
    user_transactions.sort()

    # Check if the transaction amounts are strictly increasing
    if len(user_transactions) > 1:
        for i in range(len(user_transactions) - 1):
            if user_transactions[i] >= user_transactions[i + 1]:
                return 0  # Transactions are not increasing
        return 1  # Transactions are increasing
    else:
        return 0  # Not enough transactions to determine increase


def transaction_time_fourier(user_id: str, all_transactions: list[Transaction]) -> float:
    """Performs a Fourier Transform on transaction times to detect periodicity"""
    # Initialize a list to store the user's transaction timestamps
    user_transaction_times = []

    # Collect timestamps for the specified user
    for t in all_transactions:
        if t.user_id == user_id:
            transaction_time = datetime.strptime(t.date, "%Y-%m-%d").timestamp()
            user_transaction_times.append(transaction_time)

    # Check if there are enough timestamps to perform Fourier Transform
    if len(user_transaction_times) < 3:
        return 0.0  # Not enough transactions to determine periodicity

    # Sort the user's transaction timestamps
    user_transaction_times.sort()

    # Calculate the differences between consecutive timestamps
    diffs = np.diff(user_transaction_times)

    # Perform Fourier Transform on the differences
    frequencies = fft(diffs)

    # Ensure there are enough frequencies before accessing index 1
    return float(np.abs(frequencies[1])) if len(frequencies) > 1 else 0.0


def is_large_transaction(transaction: Transaction, all_transactions: list[Transaction]) -> int:
    """Returns 1 if the transaction amount is greater than 3 times the average amount for the user."""
    # Calculate the average transaction amount for the user
    avg_amount = average_transaction_amount(transaction.user_id, all_transactions)

    # Determine if the transaction amount is greater than 3 times the average
    if transaction.amount > 3 * avg_amount:
        return 1  # Transaction is considered large
    else:
        return 0  # Transaction is not considered large


def transaction_amount_ratio(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Calculates the ratio of the transaction amount to the average amount for the user."""
    # Initialize a list to store the user's transaction amounts
    user_transactions = []

    # Collect transaction amounts for the specified user
    for t in all_transactions:
        if t.user_id == transaction.user_id:
            amount = float(t.amount)
            user_transactions.append(amount)

    # Calculate the average transaction amount for the user
    avg_amount = float(np.mean(user_transactions)) if user_transactions else 0.0

    # Calculate and return the ratio of the transaction amount to the average
    return float(transaction.amount) / avg_amount if avg_amount > 0 else 0.0


def median_transaction_interval(user_id: str, all_transactions: list[Transaction]) -> float:
    """Calculates the median time interval between transactions for a user."""
    # Initialize a list to store the user's transaction dates
    user_transaction_dates = []

    # Collect transaction dates for the specified user
    for t in all_transactions:
        if t.user_id == user_id:
            transaction_date = datetime.strptime(t.date, "%Y-%m-%d")
            user_transaction_dates.append(transaction_date)

    # Sort the user's transaction dates
    user_transaction_dates.sort()

    # Initialize a list to store intervals between transactions
    intervals = []

    # Calculate the intervals between consecutive transactions
    for i in range(1, len(user_transaction_dates)):
        interval_days = (user_transaction_dates[i] - user_transaction_dates[i - 1]).days
        intervals.append(interval_days)

    # Calculate and return the median of the intervals
    return float(np.median(intervals)) if intervals else 0.0


def normalized_transaction_frequency(user_id: str, all_transactions: list[Transaction]) -> float:
    """Calculates the normalized transaction frequency for a user."""
    # Initialize variables for total days and transaction count
    total_days = 0
    count = 0

    # Find the earliest transaction date for the specified user
    earliest_date = None

    for t in all_transactions:
        if t.user_id == user_id:
            count += 1  # Count the transaction
            transaction_date = datetime.strptime(t.date, "%Y-%m-%d")
            if earliest_date is None or transaction_date < earliest_date:
                earliest_date = transaction_date  # Update earliest date

    # Calculate total days from the earliest transaction date to now
    if earliest_date:
        total_days = (datetime.now() - earliest_date).days

    # Calculate and return the normalized transaction frequency
    return float(count / total_days) if total_days > 0 else 0.0


# def transaction_autocorrelation(user_id: str, all_transactions: list[Transaction]) -> float:
#     user_transactions = sorted(
#         [datetime.strptime(t.date, "%Y-%m-%d").timestamp() for t in all_transactions if t.user_id == user_id]
#     )

#     if len(user_transactions) < 3:
#         return 0.0  # Not enough data to calculate autocorrelation

#     diffs = np.diff(user_transactions)

#     if len(diffs) < 2:
#         return 0.0  # Not enough differences to calculate correlation

#     std_dev = np.std(diffs)

#     if std_dev == 0:
#         return 0.0  # All differences are the same, no variation to calculate correlation

#     # Calculate the correlation coefficient
#     correlation = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]

# Check if the correlation is NaN
# return correlation if not np.isnan(correlation) else 0.0

# def get_season(transaction: Transaction) -> str:
#     """Get the season for the transaction date"""
#     month = datetime.strptime(transaction.date, "%Y-%m-%d").month
#     if month in [12, 1, 2]:
#         return "Winter"
#     elif month in [3, 4, 5]:
#         return "Spring"
#     elif month in [6, 7, 8]:
#         return "Summer"
#     else:
#         return "Fall"

# def get_quarter(transaction: Transaction) -> int:
#     """Get the quarter for the transaction date"""
#     month = datetime.strptime(transaction.date, "%Y-%m-%d").month
#     return (month - 1) // 3 + 1


def get_features(transaction: Transaction, all_transactions: list[Transaction]) -> dict[str, float | int]:
    return {
        "n_transactions_same_amount": get_n_transactions_same_amount(transaction, all_transactions),
        "percent_transactions_same_amount": get_percent_transactions_same_amount(transaction, all_transactions),
        "day_of_week": get_day_of_week(transaction),
        "month": get_month(transaction),
        "day": get_day(transaction),
        "time_since_last_transaction": time_since_last_transaction(transaction, all_transactions),
        "time_since_last_transaction_same_amount": time_since_last_transaction_same_amount(
            transaction, all_transactions
        ),
        "average_transaction_amount": average_transaction_amount(transaction.user_id, all_transactions),
        "transaction_frequency": transaction_frequency(transaction.user_id, all_transactions, 30),
        "standard_deviation_of_transaction_amount": standard_deviation_of_transaction_amount(
            transaction.user_id, all_transactions
        ),
        "change_in_average_transaction_amount": change_in_average_transaction_amount(
            transaction.user_id, all_transactions, 30
        ),
        "transaction_interval_variance": transaction_interval_variance(transaction.user_id, all_transactions),
        "change_in_transaction_frequency": change_in_transaction_frequency(transaction.user_id, all_transactions, 30),
        "identical_transaction_ratio": identical_transaction_ratio(transaction, all_transactions),
        "is_start_or_end_of_month": is_start_or_end_of_month(transaction),
        "is_monthly_recurring": is_monthly_recurring(transaction, all_transactions),
        "most_common_transaction_day": most_common_transaction_day(transaction.user_id, all_transactions),
        "is_fixed_interval": is_fixed_interval(transaction, all_transactions),
        "subscription_price_consistency": subscription_price_consistency(transaction.user_id, all_transactions),
        "is_price_increasing": is_price_increasing(transaction.user_id, all_transactions),
        "transaction_time_fourier": transaction_time_fourier(transaction.user_id, all_transactions),
        "is_large_transaction": is_large_transaction(transaction, all_transactions),
        "transaction_amount_ratio": transaction_amount_ratio(transaction, all_transactions),
        "median_transaction_interval": median_transaction_interval(transaction.user_id, all_transactions),
        "normalized_transaction_frequency": normalized_transaction_frequency(transaction.user_id, all_transactions),
        # "quarter": get_quarter(transaction),
        # "season": get_season(transaction),
        # "transaction_autocorrelation": transaction_autocorrelation(transaction.user_id, all_transactions),
    }
