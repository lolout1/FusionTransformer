class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, relative_delta=0.01):
        """
        Early stopping with relative improvement threshold.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum absolute improvement (default 0)
            relative_delta: Minimum relative improvement as fraction of best loss (default 1%)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.relative_delta = relative_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Use relative threshold: improvement must be at least relative_delta % of best
            threshold = self.best_loss * (1 - self.relative_delta) - self.min_delta
            if val_loss < threshold:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True