# priority functions for joining columns with the same info

import pandas as pd

def prioritize_summary(row, left, right):
    """
    Prioritizes summaries based on conditions.

    Args:
        row (pd.Series): A row of video game data containing two summary columns.
        left (object): The name of the left summary column
        right (object): The name of the right summary column

    Returns:
        str: The longer summary or None if both summaries are missing.
    """

    left_value = row[left]
    right_value = row[right]
    if pd.isna(left_value) and not pd.isna(right_value):
        return right_value
    elif pd.isna(right_value) and not pd.isna(left_value):
        return left_value
    elif not pd.isna(left_value) and not pd.isna(right_value):
        if len(left_value) >= len(right_value):
            return left_value
        else:
            return right_value
    else:
        return None
    
def prioritize_value(row, left, right):
    """
    Prioritizes values to the larger one. Used for player counts, rating and release dates.

    Args:
        row (pd.Series): A row of video game data containing two release date columns.
        left (int, datetime): The name of the left release date column.
        right (int, datetime): The name of the right release date column.

    Returns:
        The most exact release date from the left or right columns. If both are None, returns None.
    """
    
    left_value = row[left]
    right_value = row[right]    
    if left_value and not right_value:
        return left_value
    elif right_value and not left_value:
        return right_value
    elif left_value and right_value:
        return max(left_value, right_value)
    else:
        return None
    
# join genres columns by concatenating. maybe do it in a different way later on
def join_genres(row, left, right):
    """
    Concatenates two genres columns together.

    Args:
        row (pd.Series): A row of video game data containing two genres columns.
        left (str): The name of the left genre column.
        right (str): The name of the right genre column.
    """
    left_value = row[left]
    right_value = row[right]

    if left_value and right_value:
        return left_value + ','+ right_value
    elif left_value:
        return left_value
    elif right_value:
        return right_value
    else:
        return None
    
def prioritize_columns(row, left_column, right_column):
    """
    Function to prioritize values from two columns based on certain criteria.
    
    Parameters:
        row (pandas.Series): The row of a DataFrame.
        left_column (str): The name of the left column.
        right_column (str): The name of the right column.
    
    Returns:
        The prioritized value based on the criteria.
    """
    left_value = row[left_column]
    right_value = row[right_column]
    
    # Prioritize non-empty and left over right
    if pd.notnull(left_value) and pd.notnull(right_value):
        return left_value
    elif pd.notnull(left_value):
        return left_value
    elif pd.notnull(right_value):
        return right_value
    else:
        return None