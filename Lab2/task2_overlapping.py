
from data import overlapping_data
from visual import plot_overlapping_data, svm_hard

def main():
    """A function that fits a hard-margin SVM to the overlapping data"""
    X, y = overlapping_data(n=100)
    plot_overlapping_data(X, y)
    svm_hard(X, y)

if __name__ == "__main__":
    main()
