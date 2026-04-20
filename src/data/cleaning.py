import pandas as pd


def load_data(filepath: str, drop_first_row: bool = False) -> pd.DataFrame:
    """
    Load the survey data (in either CSV format or Excel format) from a filepath.

    Parameters:
        filepath (str): The path to the survey data file.
        drop_first_row (bool): Whether to drop the first row and use the second row as column headers.

    Returns:
        pd.DataFrame: The survey data.
    """
    if filepath.endswith(".csv"):
        if drop_first_row:
            df = pd.read_csv(filepath, header=1)
        else:
            df = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        if drop_first_row:
            df = pd.read_excel(filepath, header=1)
        else:
            df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")

    return df


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
