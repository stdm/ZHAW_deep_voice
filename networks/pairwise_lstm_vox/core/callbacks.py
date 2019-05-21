from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from common.utils.paths import get_experiment_nets
from . import plot_saver as ps

# ModelCheckpoint taking active learning rounds (epoch resets) into account
# -------------------------------------------------------------------
class ActiveLearningModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, period):
        super().__init__(
            filepath=filepath,
            period=period
        )

        self.alr_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.alr_epoch += 1
        super().on_epoch_begin(self.alr_epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(self.alr_epoch, logs)


# Custom callback for own plots
# -------------------------------------------------------------------
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


# ActiveLearningLogCallback, active learning round aware epoch number logger
# -------------------------------------------------------------------
class ActiveLearningEpochLogger(Callback):
    def __init__(self, logger, total_epochs):
        super().__init__()

        self.alr_epoch = 0
        self.logger = logger
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs={}):
        self.alr_epoch += 1
        self.logger.info("Total Epoch {}/{}".format(self.alr_epoch, self.total_epochs))
