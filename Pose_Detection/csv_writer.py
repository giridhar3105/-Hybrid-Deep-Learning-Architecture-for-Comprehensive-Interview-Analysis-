import pandas as pd
from datetime import datetime

def save_to_csv(data_points):
    file_name = f"pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame(data_points)
    df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")
