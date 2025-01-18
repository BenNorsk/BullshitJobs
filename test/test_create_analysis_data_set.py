import pandas as pd
import datetime
from bullshit_jobs.data_handling._create_analysis_data_set import _create_analysis_data_set

def test_create_analysis_data_set():
    # Sample data
    data = {
        'date': [
            datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 15),
            datetime.datetime(2023, 2, 1), datetime.datetime(2023, 3, 1),
            datetime.datetime(2023, 3, 15), datetime.datetime(2023, 4, 1),
            datetime.datetime(2023, 1, 1), datetime.datetime(2023, 2, 1),
            datetime.datetime(2023, 3, 1), datetime.datetime(2023, 4, 1)
        ],
        'firm': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'outcome': [10, 20, 15, 25, 30, 35, 30, 35, 40, 45]
    }
    df = pd.DataFrame(data)

    # Parameters
    Y_col = 'outcome'
    firm = 'A'
    control_firms = ['B']
    date_col = 'date'
    treatment_date = datetime.datetime(2023, 1, 28)
    data_start_date = datetime.datetime(2023, 1, 1)
    data_end_date = datetime.datetime(2023, 4, 30)

    # Expected output
    expected_data = {
        'firm': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'treatment': [0, 1, 1, 1, 0, 0, 0, 0],
        'date': [
            datetime.datetime(2023, 1, 1), datetime.datetime(2023, 2, 1),
            datetime.datetime(2023, 3, 1), datetime.datetime(2023, 4, 1), 
            datetime.datetime(2023, 1, 1), datetime.datetime(2023, 2, 1),
            datetime.datetime(2023, 3, 1), datetime.datetime(2023, 4, 1)
        ],
        'outcome': [15.0, 15.0, 27.5, 35.0, 30.0, 35.0, 40.0, 45.0]
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["date"] = expected_df["date"].dt.to_period('M')

    # Run the function
    result_df = _create_analysis_data_set(
        df, Y_col, firm, control_firms, date_col, treatment_date, data_start_date, data_end_date
    )

    # Assertions
    # Check if the result matches the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)