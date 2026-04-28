import random

import pandas as pd


def split_records(
    records: list[dict],
    test_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Deterministic random split of records into (train, test)."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]


def load_data(filepath: str) -> tuple[pd.DataFrame, dict]:
    """Load an RCT data file with a two-row header.

    Convention:
        Row 0: short variable codes (used as DataFrame column names).
        Row 1: long-form survey questions/labels.
        Row 2+: subject responses.

    Short codes match `profile_vars`/`outcome` in config.yaml and the prompt
    JSON; the long-form labels are returned alongside so that downstream
    prompt generation can substitute the human-readable question text into
    the rendered system message.

    Parameters:
        filepath: Path to a `.csv` or `.xlsx` file following the convention above.

    Returns:
        data: DataFrame with short-code columns and only data rows.
        var_labels: Mapping from short code -> long-form label.
    """
    if filepath.endswith(".csv"):
        raw = pd.read_csv(filepath, header=0)
    elif filepath.endswith(".xlsx"):
        raw = pd.read_excel(filepath, header=0)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")

    var_labels = raw.iloc[0].to_dict()
    data = raw.iloc[1:].reset_index(drop=True)
    return data, var_labels


def include_variable_names(
    data_with_responses: pd.DataFrame, data_file_path: str
) -> pd.DataFrame:
    """Include variable names from the original data file into the provided DataFrame.

    Reads the original data file to extract column headers, maps current
    headers in the provided DataFrame back to the original headers, and
    inserts the current headers as the first row.

    Args:
        data_with_responses (pd.DataFrame): DataFrame containing the data with responses.
        data_file_path (str): Path to the original data file (CSV or XLSX) containing the headers.

    Returns:
        pd.DataFrame: DataFrame with original headers as columns and the current headers pushed to the first row.

    Raises:
        ValueError: If the provided file format is not supported (neither CSV nor XLSX).
    """

    def get_key_by_value(d, value):
        for key, val in d.items():
            if val == value:
                return key
        return value

    if data_file_path.endswith(".csv"):
        original_data_with_headers = pd.read_csv(data_file_path)
    elif data_file_path.endswith(".xlsx"):
        original_data_with_headers = pd.read_excel(data_file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")

    col_name_mapping = original_data_with_headers.iloc[0].to_dict()

    new_col_headers = []
    for col in data_with_responses.columns:
        new_col_headers.append(get_key_by_value(col_name_mapping, col))

    headers_as_first_row = pd.DataFrame(
        [data_with_responses.columns], columns=data_with_responses.columns
    )

    data_with_response_headers = pd.concat(
        [headers_as_first_row, data_with_responses], ignore_index=True
    )

    data_with_response_headers.columns = new_col_headers

    return data_with_response_headers
