import argparse

from plugins.loader import PluginLoader
from lib.utils import get_target_paths, get_folder, set_tf_allow_growth

if __name__ == '__main__':
    # TODO: swap A and B ==> from arbitrary to target
    parser = argparse.ArgumentParser(description='train swap model between A and B')
    parser.add_argument('-m', '--model-dir',
                        dest="model_dir",
                        default="model/default",
                        help="Model directory. This is where the training data will \
                            be stored. Defaults to 'model/default'")
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
    parser.add_argument('-pl', '--use-perceptual-loss',
                        action="store_true",
                        dest="perceptual_loss",
                        default=False,
                        help="Use perceptual loss while training")
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
    parser.add_argument('-c', '--cuda_visible_devices',
                        type=str,
                        default=None,
                        help="CUDA_VISIBLE_DEVICES value (e.g., -c 0,1)")

    args = parser.parse_args()
    print("Training model: {}".format(args.trainer))
    print("Training result directory: {}".format(args.model_dir))
    print('')

    if args.cuda_visible_devices is not None:
        args.gpus = len(args.cuda_visible_devices.split(','))
        if args.allow_growth:
            set_tf_allow_growth(cuda_visible_devices)

    print('Loading data, this may take a while...')
    target_image_path_dict = get_target_paths()
    print('')

    # this is so that you can enter case insensitive values for trainer
    model = PluginLoader.get_model(args.trainer)
    model = model(get_folder(args.model_dir), target_image_path_dict.keys(), args.gpus)
    model.load()
    print('')

    trainer = PluginLoader.get_trainer(args.trainer)
    trainer = trainer(model, target_image_path_dict, args.batch_size, args.perceptual_loss)
    print('')

    print('Starting training!!!')

    for epoch in range(args.epochs):

        save_iteration = epoch % args.save_interval == 0
        save_image = args.save_image and save_iteration

        trainer.train_one_step(epoch, save_image)

        if save_iteration:
            model.save_weights()