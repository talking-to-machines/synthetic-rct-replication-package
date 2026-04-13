import pandas as pd
import json
import os


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


def create_batch_file(
    prompts: pd.DataFrame,
    system_message_field: str,
    user_message_field: str = "question_prompt",
    batch_file_name: str = "batch_tasks.jsonl",
    logit: bool = False,
) -> str:
    """
    Create a JSONL batch file from the prompts DataFrame.

    Parameters:
        prompts (pd.DataFrame): The DataFrame containing prompts.
        system_message_field (str): The column name indicating the system message.
        user_message_field (str): The column name indicating the user message.
        batch_file_name (str): The name of the batch file.
        logit (bool): Whether to include logprob parameters (max_tokens=1, logprobs=True, top_logprobs=5).

    Returns:
        str: The path to the created JSONL batch file.
    """
    # Creating an array of json tasks
    tasks = []
    for i in range(len(prompts)):
        body = {
            "model": "gpt-4o-2024-08-06",  # gpt-4o-mini, gpt-4o-2024-08-06, gpt-4-turbo
            "temperature": 1.0,
            "messages": [
                {"role": "system", "content": prompts.loc[i, system_message_field]},
                {"role": "user", "content": prompts.loc[i, user_message_field]},
            ],
        }
        if logit:
            body["max_tokens"] = 1
            body["logprobs"] = True
            body["top_logprobs"] = 5
        task = {
            "custom_id": f'{prompts.loc[i, "custom_id"]}',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        tasks.append(task)

    # Creating batch file
    current_dir = os.path.dirname(__file__)
    batch_file_name = os.path.join(current_dir, f"../batch_files/{batch_file_name}")
    with open(batch_file_name, "w") as file:
        for obj in tasks:
            file.write(json.dumps(obj) + "\n")

    return batch_file_name


def include_variable_names(
    data_with_responses: pd.DataFrame, data_file_path: str
) -> pd.DataFrame:
    """Include variable names from the original data file into the provided DataFrame.
    This function reads the original data file (CSV or XLSX) to extract the column headers,
    maps the current column headers in the provided DataFrame to the original headers,
    and then inserts the current headers as the first row in the resulting DataFrame.

    Args:
        data_with_responses (pd.DataFrame): DataFrame containing the data with responses.
        data_file_path (str): Path to the original data file (CSV or XLSX) containing the headers.
    Returns:
        pd.DataFrame: DataFrame with the original headers included and the current headers as the first row.
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

    # Extract the first row from the original data
    col_name_mapping = original_data_with_headers.iloc[0].to_dict()

    new_col_headers = []
    for col in data_with_responses.columns:
        new_col_headers.append(get_key_by_value(col_name_mapping, col))

    # Push the current column headers into the first row
    headers_as_first_row = pd.DataFrame(
        [data_with_responses.columns], columns=data_with_responses.columns
    )

    # Concatenate the headers_as_first_row with the results dataframe
    data_with_response_headers = pd.concat(
        [headers_as_first_row, data_with_responses], ignore_index=True
    )

    # Assign new column headers to the results dataFrame
    data_with_response_headers.columns = new_col_headers

    return data_with_response_headers
