import os
import pandas as pd

# ================================================================
# ----------------------- LOADING HELPERS ------------------------
# ================================================================

def load_csv_from_github(
    csv_path:str,
    base_url:str="https://media.githubusercontent.com/media/d-noe/NLP_DH_PSL_Fall2025/refs/heads/main/",
):
    """
    Helper function to load a `.csv` file stored in the repository as a pandas.DataFrame.
    Originally implemented to centralize loading process, after the change to Git LFS, base URL switched from
        https://raw.githubusercontent.com/d-noe/NLP_DH_PSL_Fall2025/main/
     to https://media.githubusercontent.com/media/d-noe/NLP_DH_PSL_Fall2025/refs/heads/main/
    Input:
        - csv_path: [str]
            Path of the file within the repository (e.g.; "data/topic_data/ungdc.csv")
        - base_url: [str] | default: "https://media.githubusercontent.com/media/d-noe/NLP_DH_PSL_Fall2025/refs/heads/main/"
            Base URL for the stored files, do not mess with it.
    """
    csv_url = os.path.join(base_url,csv_path)
    df = pd.read_csv(csv_url, encoding='utf-8')
    return df