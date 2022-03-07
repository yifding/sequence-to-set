import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from args import train_argparser, eval_argparser
from config_reader import process_configs
from ssn import input_reader
from ssn.ssn_trainer import SSNTrainer


def __train(run_args):
    trainer = SSNTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SSNTrainer(run_args)
    # trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
    #              input_reader_cls=input_reader.JsonInputReader)

    print(f'run_args: {run_args}')
    os.makedirs(run_args.yd_output_dir, exist_ok=True)
    for yd_test_att in run_args.yd_test_att_list:
        trainer._predictions_path = os.path.join(run_args.yd_output_dir, f'{yd_test_att}.json')
        test_dataset_path = os.path.join(run_args.yd_test_dir, f'{yd_test_att}.json')
        trainer.eval(dataset_path=test_dataset_path, types_path=run_args.types_path,
                     input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    print(args.mode)

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python ssn.py train ...'")
