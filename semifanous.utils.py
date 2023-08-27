import pandas as pd


# save the ground truth for semi-famous -> item_base == d_3_*

def get_semi_famous_ground_truth(all_data_csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(all_data_csv_path)
    
    semi_famous_data = df[df['item_base'].str.startswith('d_3_')]
    
    # Only keep 'item_base' and 'name' columns
    return semi_famous_data[['item_base', 'name']]


full_data_csv_path = f"textwash_data/study1/intruder_test/full_data_study.csv"
semi_famous_ground_truth = get_semi_famous_ground_truth(full_data_csv_path)
semi_famous_ground_truth.to_csv("semi_famous_ground_truth.csv", index=False)
