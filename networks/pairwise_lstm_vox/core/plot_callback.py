from keras.callbacks import Callback
from . import plot_saver as ps

class PlotCallback(Callback):
    def __init__(self, network_name, reset_train_begin=False):
        super().__init__()
        self.network_name = network_name
        self.reset_train_begin = reset_train_begin
        self.reset()

    def reset(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    def on_train_begin(self, logs={}):
        if self.reset_train_begin:
            self.reset()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1
        
        ps.save_accuracy_plot_direct(self.network_name, self.acc, self.val_acc)
        ps.save_loss_plot_direct(self.network_name, self.losses, self.val_losses)
