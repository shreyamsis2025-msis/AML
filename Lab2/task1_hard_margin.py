
from data import generate_linear_data
from visual import hard_margin_svm_visualization

def main():
    """A function that fits a hard-margin SVM to the data"""
    X, y = generate_linear_data(n=100)
    hard_margin_svm_visualization(X, y)
     


if __name__ == "__main__":
    main()
