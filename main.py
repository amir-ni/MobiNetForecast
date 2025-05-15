import argparse
import MobiNetForecast.utils as utils
from MobiNetForecast.config_loader import load_config, expand_config


def main() -> None:
    """
    Main function to handle argument parsing and execute training or testing.
    """
    parser = argparse.ArgumentParser(description='Trajectory Prediction Learning')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--baseline', type=str, required=False, help='Baseline model to run')
    parser.add_argument('--test', default=False, action='store_true', help='Evaluate the model')
    parser.add_argument('--search', default=False, action='store_true', help='Perform grid search')
    args = parser.parse_args()

    config_list = load_config(args.config)

    if args.search:
        config_list = expand_config(config_list)
        search_dict = dict()

    for name, config in config_list.items():
        utils.setup_environment(config["seed"])
        dataset = utils.get_dataset(config, test_mode=not args.search and args.test)
        model = utils.select_model(args.baseline, config)

        if args.search:
            utils.train_model(name, dataset, config, model)
            dataset = utils.get_dataset(config, test_mode=True)
            search_dict[name] = utils.test_model(name, dataset, config, model)
        elif args.test:
            utils.test_model(name, dataset, config, model)
        else:
            utils.train_model(name, dataset, config, model)

    if args.search:
        utils.save_search(search_dict)


if __name__ == '__main__':
    main()
