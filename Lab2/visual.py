import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from data import overlapping_data

def plot_2d_data(X,y,title="Linearly Separable Data"):
    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.show()

def hard_margin_svm_visualization(X, y):
    model = SVC(C=1e6, kernel='linear')  # Large C for hard margin
    model.fit(X, y)
    w=model.coef_[0] #weights
    b=model.intercept_[0] #bais
    
    plt.figure(figsize=(7,6))
    
    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',edgecolors='k')
    
    plt.scatter(
        model.support_vectors_[:,0],
        model.support_vectors_[:,1],
        s=120,facecolors='none',edgecolors='k',label='Support Vectors'
    )
    
    x_vals=np.linspace(X[:,0].min()-1,X[:,0].max()+1,200)
    
    y_decision = -(w[0]*x_vals + b)/w[1]
    
    y_margin_pos = -(w[0]*x_vals + b - 1)/w[1]
    y_margin_neg = -(w[0]*x_vals + b + 1)/w[1]
    
    plt.plot(x_vals,y_decision,'k-',label='Decision Boundary')
    plt.plot(x_vals,y_margin_pos,'k--',label='Margin +1')
    plt.plot(x_vals,y_margin_neg,'k--',label='Margin -1')
    
    plt.fill_between(x_vals,y_margin_pos,y_margin_neg,color='grey',alpha=0.2,label='Margin Area')
    
    plt.title("Hard-Margin SVM with Margin Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
    

def plot_overlapping_data(X,y):
    X_overlap,y_overlap=overlapping_data()
    plt.scatter(X_overlap[:,0], X_overlap[:,1], c=y_overlap, cmap='bwr')
    plt.title('Overlapping Data')
    plt.show()
    
def svm_hard(X_overlap,y_overlap):
        
    svm_hard=SVC(C=1e6,kernel='linear')
    svm_hard.fit(X_overlap,y_overlap)
    
    w=svm_hard.coef_[0]
    b=svm_hard.intercept_[0]
    print("Weight vector (w):", w)
    print("Bias (b):", b)
    
    plt.figure(figsize=(7,6))
    
    plt.scatter(X_overlap[:,0],X_overlap[:,1],c=y_overlap,cmap='bwr',edgecolors='k')
    
    plt.scatter(
        svm_hard.support_vectors_[:,0],
        svm_hard.support_vectors_[:,1],
        s=120,facecolors='none',edgecolors='k',label='Support Vectors'
    )

    x_vals=np.linspace(X_overlap[:,0].min()-1,X_overlap[:,0].max()+1,200)
    
    y_decision = -(w[0]*x_vals + b)/w[1]
    
    y_margin_pos = -(w[0]*x_vals + b - 1)/w[1]
    y_margin_neg = -(w[0]*x_vals + b + 1)/w[1]
    
    plt.plot(x_vals,y_decision,'k-',label='Decision Boundary')
    plt.plot(x_vals,y_margin_pos,'k--',label='Margin +1')
    plt.plot(x_vals,y_margin_neg,'k--',label='Margin -1')
    
    plt.fill_between(x_vals,y_margin_pos,y_margin_neg,color='grey',alpha=0.2,label='Margin Area')
    
    plt.title("Hard-Margin SVM with Margin Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
    

def soft_margin(X,y,c):
    model = SVC(C=c, kernel='linear')  # Soft margin with C=1
    model.fit(X, y)
    w=model.coef_[0]
    b=model.intercept_[0]
    
    print("Weight vector (w):", w)
    print("Bias (b):", b)
    
    plt.figure(figsize=(7,6))
    
    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',edgecolors='k')
    
    plt.scatter(
        model.support_vectors_[:,0],
        model.support_vectors_[:,1],
        s=120,facecolors='none',edgecolors='k',label='Support Vectors'
    )
    
    x_vals=np.linspace(X[:,0].min()-1,X[:,0].max()+1,200)
    
    y_decision = -(w[0]*x_vals + b)/w[1]
    
    y_margin_pos = -(w[0]*x_vals + b - 1)/w[1]
    y_margin_neg = -(w[0]*x_vals + b + 1)/w[1]
    
    plt.plot(x_vals,y_decision,'k-',label='Decision Boundary')
    plt.plot(x_vals,y_margin_pos,'k--',label='Margin +1')
    plt.plot(x_vals,y_margin_neg,'k--',label='Margin -1')
    
    plt.fill_between(x_vals,y_margin_pos,y_margin_neg,color='grey',alpha=0.2,label='Margin Area')
    
    plt.title("Soft-Margin SVM with Margin Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()