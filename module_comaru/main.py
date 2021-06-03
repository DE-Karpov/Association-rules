import argparse
import pandas as pd

from service.rules_controller import get_all_common_rules, get_all_assoc_rules


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '-p', '--path', help='Set absolute path to your dataset', required=True)
    parser.add_argument('-z', '--zone', default='union',
                        help='Set zone param [union | intersection]. Default: union')
    parser.add_argument('-s', '--support', default=0.1,
                        help='Set minimum support param. Default: 0.1')
    parser.add_argument('-c', '--confidence', default=0.1,
                        help='Set minimum confidence param. Default: 0.1')
    args = parser.parse_args()
    return args


def get_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print('FileNotFound error. There is no file with absolute path ' + path)


def save_to_file(result):
    with open('result.txt', 'w') as f:
        for row in result:
            for item in row:
                f.write(item + ' ')
            f.write('\n')


def get_rules(dataset, args: argparse.Namespace) -> None:
    try:
        result = get_all_common_rules(dataset, args) if args.zone == 'intersection' else get_all_assoc_rules(dataset, args)
        save_to_file(result)
        print("Success!")
    except Exception as e:
        raise Exception("Something went wrong!") from e
        pass


if __name__ == '__main__':
    args = handle_args()
    dataset = get_data(args.path)
    get_rules(dataset, args)
