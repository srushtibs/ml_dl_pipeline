import pandas as pd
import matplotlib.pyplot as plt

classical = pd.read_csv("classical_results.csv")
rnn = pd.read_csv("rnn_imdb_results.csv")
cnn = pd.read_csv("cnn_cifar10_results.csv")

classical["Dataset"] = "IMDB"
rnn["Dataset"] = "IMDB"
cnn["Dataset"] = "CIFAR-10"

classical["Model_Type"] = "Classical ML"
rnn["Model_Type"] = "Deep Learning"
cnn["Model_Type"] = "Deep Learning"

all_results = pd.concat([classical, rnn, cnn], ignore_index=True)

all_results.to_csv("final_results.csv", index=False)
print(" Combined results saved to final_results.csv")

plt.figure(figsize=(8, 5))
for dataset in all_results["Dataset"].unique():
    subset = all_results[all_results["Dataset"] == dataset]
    plt.bar(subset["Model"], subset["Accuracy"], label=dataset)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

plt.figure(figsize=(8, 5))
for dataset in all_results["Dataset"].unique():
    subset = all_results[all_results["Dataset"] == dataset]
    plt.bar(subset["Model"], subset["F1 Score"], label=dataset)

plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.title("Model F1 Score Comparison")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("f1_comparison.png")
plt.show()