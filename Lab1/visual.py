import matplotlib.pyplot as plt

def plot_2d_data(X,y,title="linear Data"):
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title(title)
    plt.xlabel("x1")
    plt.xlabel("y1")
    plt.show()

# def plot_2d_data1(X,y,title="Non linear Data"):
#     plt.scatter(X[:,0]>0,X[:,1]>0,c=y)
#     plt.title(title)
#     plt.xlabel("x1")
#     plt.xlabel("y1")
#     plt.show()