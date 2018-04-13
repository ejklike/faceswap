import argparse

from plugins.loader import PluginLoader
from lib.utils import get_image_paths, get_folder


def set_tf_allow_growth(self):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

if __name__ == '__main__':
    # TODO: swap A and B ==> from arbitrary to target
    parser = argparse.ArgumentParser(description='train swap model between A and B')
    parser.add_argument('-A', '--input-A',
                        dest="input_A",
                        default="data_A",
                        help="Input directory. A directory containing training images for face A.\
                            Defaults to 'input'")
    parser.add_argument('-B', '--input-B',
                        dest="input_B",
                        default="data_B",
                        help="Input directory. A directory containing training images for face B.\
                            Defaults to 'input'")
    parser.add_argument('-m', '--model-dir',
                        dest="model_dir",
                        default="results/model",
                        help="Model directory. This is where the training data will \
                            be stored. Defaults to 'model'")
    parser.add_argument('-s', '--save-interval',
                        type=int,
                        dest="save_interval",
                        default=100,
                        help="Sets the number of iterations before saving the model.")
    parser.add_argument('-si', '--save-image',
                        action="store_true",
                        dest="save_image",
                        default=False,
                        help="Sets save_image option to save current model results")
    parser.add_argument('-t', '--trainer',
                        type=str,
                        choices=PluginLoader.get_available_models(),
                        default=PluginLoader.get_default_model(),
                        help="Select which trainer to use.")
    parser.add_argument('-bs', '--batch-size',
                        type=int,
                        default=64,
                        help="Batch size, as a power of 2 (64, 128, 256, etc)")
    parser.add_argument('-ag', '--allow-growth',
                        action="store_true",
                        dest="allow_growth",
                        default=False,
                        help="Sets allow_growth option of Tensorflow to spare memory on some configs")
    parser.add_argument('-ep', '--epochs',
                        type=int,
                        default=1000000,
                        help="Length of training in epochs.")
    parser.add_argument('-g', '--gpus',
                        type=int,
                        default=1,
                        help="Number of GPUs to use for training")

    args = parser.parse_args()

    print("Data A Directory: {}".format(args.input_A))
    print("Data B Directory: {}".format(args.input_B))
    print("Training result directory: {}".format(args.model_dir))
    print('')

    if args.allow_growth:
        set_tf_allow_growth()

    print('Loading data, this may take a while...')
    images_A = get_image_paths(args.input_A)
    images_B = get_image_paths(args.input_B)
    print('')

    # this is so that you can enter case insensitive values for trainer
    model = PluginLoader.get_model(args.trainer)(get_folder(args.model_dir), args.gpus)
    model.load_weights()
    print('')

    trainer = PluginLoader.get_trainer(args.trainer)
    trainer = trainer(model, images_A, images_B, args.batch_size)
    print('')

    print('Starting training!!!')

    for epoch in range(args.epochs):

        save_iteration = epoch % args.save_interval == 0
        save_image = args.save_image and save_iteration

        trainer.train_one_step(epoch, save_image)

        if save_iteration:
            model.save_weights()