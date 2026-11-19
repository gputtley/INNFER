"""Era selections must keep the correct binned categories and fit namespaces."""
import ast
import argparse
import copy
import fnmatch
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


def functions():
    tree = ast.parse((ROOT / 'python/worker/useful_functions.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ['GetCategoryLoop', 'GetSnakeMakeStepLoop']]
    namespace = {'copy': copy, 'fnmatch': fnmatch, 'yaml': yaml}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<repository source>', 'exec'), namespace)
    return namespace


class FSRWorkflowCategoryTests(unittest.TestCase):
    def test_directory_suffix_survives_generated_command_round_trip(self):
        main = next(n for n in ast.parse((ROOT / 'scripts/innfer.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        # Execute the real directory-option normalization at entry to main.
        normalization = ast.fix_missing_locations(ast.Module(body=main.body[:2], type_ignores=[]))
        module_class = next(n for n in ast.parse((ROOT / 'python/worker/module.py').read_text()).body
                            if isinstance(n, ast.ClassDef) and n.name == 'Module')
        serialize = next(n for n in module_class.body if isinstance(n, ast.FunctionDef)
                         and n.name == '_MakeCommandOptions')
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[serialize], type_ignores=[])),
                     '<command serialization>', 'exec'), namespace)
        parser = argparse.ArgumentParser()
        for option in ['extra-dir-name', 'extra-extra-dir-name', 'extra-input-dir-name', 'extra-output-dir-name']:
            parser.add_argument('--'+option, default='')
        defaults = parser.parse_args([])
        for base, output in [('', ''), ('', 'PostFit'), ('Asimov', '')]:
            with self.subTest(base=base, output=output):
                args = parser.parse_args([])
                args.extra_dir_name = base
                args.extra_extra_dir_name = '_run2'
                args.extra_output_dir_name = output
                exec(compile(normalization, '<directory normalization>', 'exec'), {'args': args})
                expected = (args.extra_input_dir_name, args.extra_output_dir_name)
                options = namespace['_MakeCommandOptions'](SimpleNamespace(default_args=defaults), args)
                runtime = parser.parse_args(shlex.split(' '.join(options)))
                exec(compile(normalization, '<directory normalization>', 'exec'), {'args': runtime})
                self.assertEqual((runtime.extra_input_dir_name, runtime.extra_output_dir_name), expected)

    def test_exact_and_wildcard_selections_keep_config_order_and_no_duplicates(self):
        get = functions()['GetCategoryLoop']
        cfg = {'categories': dict.fromkeys(['run2_a', '2223_a', 'run2_b', '24_a'])}
        self.assertEqual(get(cfg), list(cfg['categories']))
        self.assertEqual(get(cfg, ['run2_*']), ['run2_a', 'run2_b'])
        self.assertEqual(get(cfg, ['run2_*', 'run2_a', '24_a']), ['run2_a', 'run2_b', '24_a'])
        self.assertEqual(get(cfg, '2223_*,24_a'), ['2223_a', '24_a'])
        self.assertEqual(get(cfg, ['run2_a']), ['run2_a'])
        self.assertEqual(get(cfg, []), [])
        self.assertEqual(get({}, ['run2_*']), ['inclusive'])

    def test_workflow_expands_four_complete_fit_chains_with_independent_suffixes(self):
        namespace = functions()
        workflow = yaml.safe_load((ROOT / 'configs/snakemake/btm_binned_fsr_condor_ic.yaml').read_text())
        current = Path.cwd()
        try:
            os.chdir(ROOT)
            steps = namespace['GetSnakeMakeStepLoop'](workflow)
        finally:
            os.chdir(current)
        # The full workflow also contains preprocessing and prefit plots.
        steps = steps[next(i for i, step in enumerate(steps)
                           if step['step'] == 'InitialFit'):]
        chain = ['InitialFit', 'HessianNumerical', 'Covariance', 'ScanPointsFromHessian',
                 'Scan', 'ScanCollect', 'ScanPlot', 'DistributionPlot']
        self.assertEqual([s['step'] for s in steps], chain*4)
        for index, era in enumerate([None, 'run2', '2223', '24']):
            for step in steps[index*8:(index+1)*8]:
                options = step['run_options']
                self.assertEqual(options.get('specific_category'), None if era is None else era+'_*')
                suffix = options.get('extra_extra_dir_name', '')
                self.assertEqual(suffix, '' if era is None else '_'+era)
                # These are the transformations performed by innfer.main.
                self.assertEqual('InitialFit'+suffix,
                                 'InitialFit' if era is None else 'InitialFit_'+era)
                if step['step'] == 'DistributionPlot':
                    self.assertEqual(options['extra_output_dir_name']+suffix,
                                     'PostFit' if era is None else 'PostFit_'+era)
                self.assertEqual(options['data_type'], 'data')


if __name__ == '__main__':
    unittest.main()
