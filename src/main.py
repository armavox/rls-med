import warnings
import utils.helpers as H


warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(config):
    pass


if __name__ == "__main__":
    args = H.arguments()
    config = H.load_params_namespace(args.train_config)
    H.set_logging_config("./configs/logging_config.yaml", config.metaconf["ws_path"])

    use_cuda = (True if config.metaconf["ngpus"] != 0 else False)
    H.random_seed_init(config.trainer["random_seed"], use_cuda)

    run(config)
