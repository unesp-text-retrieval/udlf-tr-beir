import pandas as pd
import time


class CommonOutput:
    def dataframe(self, method: str = None):
        # Flatten the nested dictionaries
        environment = f"{self.dataset.dataset_name}+{self.name}{'+' + method if method else ''}"
        metrics = ['ndcg', 'map', 'recall', 'precision']
        rows = []
        for metric in metrics:
            for key, value in self.data[metric].items():
                rows.append({
                    'Metric': metric.upper(),
                    'K': key.split('@')[1],  # Extract the K value from the key
                    'Value': value,
                    'Environment': environment
                })

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows)

        # Pivot the DataFrame to make K values columns
        df_pivot = df.pivot(index=['Metric', 'Environment'], columns='K', values='Value')
        df_pivot.columns = [f'K{k}' for k in df_pivot.columns]  # Rename columns to include 'K'
        df_pivot.reset_index(inplace=True)  # Reset index to make 'Metric' and 'Model' columns

        return df_pivot