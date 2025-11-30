import pandas as pd

def load_data(filename="Energy_consumption.csv"):
    """Load CSV dataset from the repo root."""
    # Get repo root (one level up from src)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(repo_root, filename)
    
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Clean and encode dataset for modeling."""
    # Map DayOfWeek to numeric
    week_map = {'Monday':1, 'Tuesday':2, 'Wednesday':3,
                'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    df['DayOfWeek'] = df['DayOfWeek'].map(week_map)

    # Map categorical features
    map1 = {'On':1, 'Off':0}
    map2 = {'No':0, 'Yes':1}
    df['HVACUsage'] = df['HVACUsage'].map(map1)
    df['LightingUsage'] = df['LightingUsage'].map(map1)
    df['Holiday'] = df['Holiday'].map(map2)

    # Drop unnecessary columns
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    return df
