from matplotlib import pyplot as plt

def draw_loss(loss,val_loss):
    plot1,=plt.plot(range(1,len(loss)+1),loss,color='g',marker='^')
    plot2,=plt.plot(range(1,len(val_loss)+1),val_loss,color='r',linestyle='--',marker='o')
    plt.xlabel("epochs",color='c')
    plt.ylabel("loss",color='c')
    plt.yticks(rotation=30)
    plt.ylim(0,1)
    plt.locator_params('y',nbins=20)
    y2=plt.twinx()
    y2.set_ylim(0,1)
    y2.locator_params('y',nbins=20)
    plt.grid()
    plt.legend([plot1, plot2],["loss", "val_loss"],loc="upper right")
    plt.title("loss and val_loss")
    plt.show()
