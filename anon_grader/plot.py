import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot a scatter plot of the predictions of the few shot models vs human rate

# Read the prediction of the fewshot model
predictions = pd.read_csv(
    "./anon_grader/results/predictions_1_famous_w_few_shot.csv", index_col=0
)

# Sort the DataFrame by gold labels
sorted_df = predictions.sort_values(by="human_rate")

# Create the scatter plot
x = np.arange(len(sorted_df))
plt.figure(figsize=(10, 6))
plt.scatter(x, sorted_df["human_rate"], label="human_rate", alpha=1, marker="x")
plt.scatter(x, sorted_df["RoBERTa"], label="RoBERTa", alpha=0.5)
# plt.scatter(x, sorted_df['zero_shot'], label='zero_shot', alpha=0.5)
# plt.scatter(x, sorted_df['multi_persona'], label='multi_persona', alpha=0.5)
plt.scatter(x, sorted_df["one_shot_0"], label="one_shot_0", alpha=0.5)
# plt.scatter(x, sorted_df['one_shot_1'], label='one_shot_1', alpha=0.5)
# plt.scatter(x, sorted_df['three_shot'], label='three_shot', alpha=0.5)
# plt.scatter(x, sorted_df['CoT'], label='CoT', alpha=0.5)
# plt.scatter(x, sorted_df['self_const_zero_shot'], label='self_const_zero_shot', alpha=0.5)
# plt.scatter(x, sorted_df['self_const_three_shot'], label='self_const_three_shot', alpha=0.5)


# Customize the plot
plt.title("Regression Task: Predictions vs Human Rates")
plt.xlabel("Human Rates")
plt.ylabel("Model Predictions")
plt.legend()
plt.xticks([])
plt.grid(True)

# Save the plot
plt.savefig("./anon_grader/results/predictions_vs_human_rates.png")
