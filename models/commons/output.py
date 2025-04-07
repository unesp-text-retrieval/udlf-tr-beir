import pandas as pd
import time


class CommonOutput:
    @property
    def dataframe(self):
        # Flatten the nested dictionaries
        metrics = ['ndcg', 'map', 'recall', 'precision']
        rows = []
        for metric in metrics:
            for key, value in self.data[metric].items():
                rows.append({
                    'Metric': metric.upper(),
                    'K': key.split('@')[1],  # Extract the K value from the key
                    'Value': value,
                    'Environment': f"{self.dataset_name}+{self.name}"  # Combine dataset name and model name
                })

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows)

        # Pivot the DataFrame to make K values columns
        df_pivot = df.pivot(index=['Metric', 'Environment'], columns='K', values='Value')
        df_pivot.columns = [f'K{k}' for k in df_pivot.columns]  # Rename columns to include 'K'
        df_pivot.reset_index(inplace=True)  # Reset index to make 'Metric' and 'Model' columns

        return df_pivot