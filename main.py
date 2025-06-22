"""Configuration Module."""

from os import path

base_dir = path.abspath(path.dirname(__file__))
log_dir = path.join(base_dir, 'logs')
run_dir = path.join(base_dir, 'runs')
data_dir = path.join(base_dir, 'data')
res_dir = path.join(base_dir, 'data', 'benchmark')
conf_path = path.join(base_dir, 'config', 'config.yaml')
