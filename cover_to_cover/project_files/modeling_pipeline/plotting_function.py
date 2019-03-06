import matplotlib.pyplot as plt
import pandas as pd
def plot_model_history(csv_file, model_name, data):
    df = pd.read_csv(csv_file)
    train_loss = df['loss']
    val_loss   = df['val_loss']
    train_acc  = df['acc']
    val_acc    = df['val_acc']
    xc         = range(df.shape[0])
    title = "{}: {}".format(model_name, data)
    plt.figure()
    plt.suptitle(title[:-4], fontweight='bold')
    plt.title('Accuracy')
    
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('Epoch')
    plt.legend(['training', 'test'])

    plt.figure()
    plt.suptitle(title[:-4], fontweight='bold')
    plt.title('Loss')
    
    plt.ylim(top=1)
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('Epoch')
    plt.legend(['training', 'test'])
    plt.show()