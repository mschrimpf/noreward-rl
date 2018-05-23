import os


class Dataset(object):

    def __init__(self):
        # # #
        # Dataset general
        self.dataset_path = ""
        self.proportion_training_set = 0.95
        self.shuffle_data = True

        # # #
        # For reusing tfrecords:
        self.reuse_TFrecords = False
        self.reuse_TFrecords_ID = 0
        self.reuse_TFrecords_path = ""

        # # #
        # Set random labels
        self.random_labels = False
        self.scramble_data = False

        # # #
        # Transfer learning
        self.transfer_learning = False
        self.transfer_pretrain = True
        self.transfer_label_offset = 0
        self.transfer_restart_name = "_pretrain"
        self.transfer_append_name = ""

        # Find dataset path:
        self.dataset_path = '../data/cifar10/'

    # # #
    # Dataset general
    # Set base tfrecords
    def generate_base_tfrecords(self):
        self.reuse_TFrecords = False

    # Set reuse tfrecords mode
    def reuse_tfrecords(self, experiment):
        self.reuse_TFrecords = True
        self.reuse_TFrecords_ID = experiment.ID
        self.reuse_TFrecords_path = experiment.name

    # # #
    # Transfer learning
    def do_pretrain_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_append_name = self.transfer_restart_name

    def do_transfer_transfer_lerning(self):
        self.transfer_learning = True
        self.transfer_pretrain = True
        self.transfer_label_offset = 5
        self.transfer_append_name = "_transfer"


class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.num_epochs_per_decay = 1.0
        self.learning_rate_factor_per_decay = 0.95
        self.weight_decay = 0
        self.max_num_epochs = 60
        self.crop_size = 28
        self.image_size = 32
        self.drop_train = 1
        self.drop_test = 1
        self.momentum = 0.9
        self.augmentation = False


class Experiment(object):

    def __init__(self, id=0, name='base'):
        self.name = "base"
        self.log_dir_base = os.path.join(os.path.dirname(__file__), '..', 'log-42_42_1')

        # Recordings
        self.max_to_keep_checkpoints = 5
        self.recordings = False
        self.num_batches_recordings = 0

        # Plotting
        self.plot_details = 0
        self.plotting = False

        # Test after training:
        self.test = False

        # Start from scratch even if it existed
        self.restart = False

        # Skip running experiments
        self.skip = False

        # Save extense summary
        self.extense_summary = True

        # Add ID to name:
        self.ID = id
        self.name = 'ID' + str(self.ID) + "_" + name

        # Add additional descriptors to Experiments
        self.dataset = Dataset()
        self.hyper = Hyperparameters()

    def __repr__(self):
        return str(self.get_vars(self))

    def get_vars(self, obj):
        obj_vars = vars(obj)
        for key, value in obj_vars.items():
            if hasattr(value, '__dict__'):
                obj_vars[key] = self.get_vars(value)
        return obj_vars

    def do_recordings(self, max_epochs):
        self.max_to_keep_checkpoints = 0
        self.recordings = True
        self.hyper.max_num_epochs = max_epochs
        self.num_batches_recordings = 10

    def do_plotting(self, plot_details=0):
        self.plot_details = plot_details
        self.plotting = True
