import pandas as pd
from pathlib import Path
from src.utils.common import read_params

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    source_path = params['data_ingestion']['source_path']
    output_file = params['data_ingestion']['output_file']

    # Mock Data Creation for demonstration purposes if the source CSV is missing
    if not Path(source_path).exists():
        print(f"Warning: Mocking data as source file not found at {source_path}.")
        data = pd.DataFrame({
            'Tectonic setting': ['MOR', 'OIB', 'SSZ', 'MOR', 'OIB', 'SSZ'],
            'SiO2': [45, 48, 51, 46, 49, 50],
            'MgO': [8, 7, 6, 7.5, 6.5, 5.5],
            'Al2O3': [15, 16, 17, 15.5, 16.5, 17.5],
            'Col_1': range(6), 'Sample': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'] 
        })
    else:
        try:
            data = pd.read_csv(source_path)
        except FileNotFoundError:
            print(f"Error: Source file not found at {source_path}. Please check config.")
            return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Data Ingestion complete: {output_path}")

if __name__ == "__main__":
    main()
