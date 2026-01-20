
from data import overlapping_data
from visual import soft_margin

def main():
    """A function that fits a soft-margin SVM to the overlapping data"""
    X, y = overlapping_data(n=100)
    # plot_overlapping_data(X, y)
    for i in [0.01, 0.1, 1.0, 10.0, 100.0]:
        soft_margin(X, y, c=i)

if __name__ == "__main__":
    main()
