#!/usr/bin/env python
import sys
import yaml
import argparse


class ConfigParser(object):
    def __init__(self, *pargs, **kwpargs):
        self.options = []
        self.pargs = pargs
        self.kwpargs = kwpargs

    def add(self, *args, **kwargs):
        self.options.append((args, kwargs))

    def parse(self, args=None):
        if args is None:
            args = sys.argv[1:]

        conf_parser = argparse.ArgumentParser(add_help=False)
        conf_parser.add_argument("-c", "--config",
                                 default=None,
                                 help="where to load YAML configuration",
                                 metavar="FILE")

        res, remaining_argv = conf_parser.parse_known_args(args)

        config_vars = {}
        if res.config is not None:
            with open(res.config, 'r') as stream:
                config_vars = yaml.load(stream)

        parser = argparse.ArgumentParser(
            *self.pargs,
            # Inherit options from config_parser
            parents=[conf_parser],
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **self.kwpargs,
        )

        for opt_args, opt_kwargs in self.options:
            parser_arg = parser.add_argument(*opt_args, **opt_kwargs)
            if parser_arg.dest in config_vars:
                config_default = config_vars.pop(parser_arg.dest)
                expected_type = str
                if parser_arg.type is not None:
                    expected_type = parser_arg.type

                if not isinstance(config_default, expected_type):
                    parser.error('YAML configuration entry {} '
                                 'does not have type {}'.format(
                                     parser_arg.dest,
                                     expected_type))

                parser_arg.default = config_default

        if config_vars:
            parser.error('unexpected configuration entries: ' + \
                         ', '.join(config_vars))

        return parser.parse_args(remaining_argv)


if __name__ == "__main__":
    cp = ConfigParser()

    cp.add("python_exec")
    cp.add("file")
    cp.add("memory", type=int)
    cp.add("cpus", type=int)
    cp.add("parameters")
    print(cp.parse())

# yaml_file = sys.argv[1]
# with open(yaml_file, 'r') as stream:
#     try:
#         command = yaml.safe_load(stream)
#     except yaml.YAMLError as exc:
#         print(exc)

#     python_env = command["env"]
#     file = command["file"]
#     batch_job_name = command["batch_job_name"]
#     memory = command["memory"]
#     cpus = command["cpus"]
#     parameters:
#     - / homes / jerphanion / mesh - processing - pipeline / data / original / MNS_M897_115.tif
#     - / homes / jerphanion / mesh - processing - pipeline / out / crf
#
# options:
# - max_iter: [2]
# - std_pos: [3.0]
# - weight_pos: [10.0]
# - std_bilat: [3.0]
# - weight_bilat: [15.0]