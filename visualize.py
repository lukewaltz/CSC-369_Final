import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the uploaded CSV file to examine its contents
file_path = 'predictions.csv'
data = pd.read_csv(file_path)

# Separate actual and prediction values
actual = data['Actual']
predicted = data['Prediction']

# Create a confusion matrix
def confusion_matrix_plot():
    conf_matrix = confusion_matrix(actual, predicted)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')

# Generate and display classification metrics
def classification_metrics_plot():
    # Generate classification report
    report = classification_report(actual, predicted, target_names=["Class 0", "Class 1"])
    
    # Save the classification report to a file
    with open("classification_report.txt", "w") as file:
        file.write("Classification Report:\n")
        file.write(report)
    
    # Scatter plot to compare actual vs prediction
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data.index, y=actual, label='Actual', color='blue', alpha=0.6)
    sns.scatterplot(x=data.index, y=predicted, label='Prediction', color='red', alpha=0.6)
    plt.title("Actual vs Prediction Scatter Plot")
    plt.xlabel("Index")
    plt.ylabel("Class")
    plt.legend()
    plt.savefig('scatter_plot.png')  # Save the plot as a PNG file

# Calculate evaluation metrics
def evaluation_metrics():
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    # Collect the metrics
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    metrics_df = pd.DataFrame(metrics, index=["Score"])

    # Write metrics to a file
    with open("evaluation_metrics.txt", "w") as file:
        file.write("Evaluation Metrics:\n")
        file.write(metrics_df.to_string())

def k_accuracy():
    # Read the existing k_accuracy.csv file
    df = pd.read_csv('k_accuracy.csv')

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Create a line plot with markers
    plt.plot(df['k'], df['Accuracy'], marker='o', linestyle='-', linewidth=2, markersize=8)

    # Customize the plot
    plt.title('Prediction Accuracy vs. K Value', fontsize=16)
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate the highest accuracy point
    max_accuracy = df.loc[df['Accuracy'].idxmax()]
    plt.annotate(f'Highest Accuracy: {max_accuracy["Accuracy"]}%\nAt K = {max_accuracy["k"]}', 
                xy=(max_accuracy['k'], max_accuracy['Accuracy']),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", alpha=0.3))

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('k_accuracy_plot.png')


# Call the functions to generate the visualizations and metrics
confusion_matrix_plot()
classification_metrics_plot()
evaluation_metrics()
k_accuracy()
