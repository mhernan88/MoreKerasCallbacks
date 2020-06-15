from keras.callbacks import ModelCheckpoint
import warnings


class S3Checkpoint(ModelCheckpoint):
    def __init__(self, filepath, s3_client, s3_bucket, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, s3_filepath=None, *args, **kwargs):
        super(S3Checkpoint, self).__init__(
            filepath, monitor, verbose, save_best_only,
            save_weights_only, mode, period, *args, **kwargs)
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_filepath = s3_filepath

    def upload_to_s3(self, filepath):
        if self.s3_filepath is None:
            s3_filepath = filepath
        else:
            s3_filepath = self.s3_filepath
        _ = self.s3_client.upload_file(filepath, self.s3_bucket, s3_filepath)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                            self.upload_to_s3(filepath)
                        else:
                            self.model.save(filepath, overwrite=True)
                            self.upload_to_s3(filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                    self.upload_to_s3(filepath)
                else:
                    self.model.save(filepath, overwrite=True)
                    self.upload_to_s3(filepath)