import argparse
import copy
import os
import sys
from functools import partial

import numpy as np
import wandb
import yaml
from other_functions import GetValidateLoop, GetPOILoop, GetNuisanceLoop, GetCombinedValidateLoop, GetYName, \
    MakeYieldFunction
from plotting import plot_likelihood, plot_stacked_histogram_with_ratio

print("Running INNFER")

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='Config for running', default=None)
parser.add_argument('--benchmark', help='Run from benchmark scenario', default=None,
                    choices=["Gaussian", "GaussianWithExpBkg", "GaussianWithExpBkgVaryingYield"])
parser.add_argument('--architecture', help='Config for running', default="configs/architecture/default.yaml")
parser.add_argument('--submit', help='Batch to submit to', type=str, default=None)
parser.add_argument('--step', help='Step to run', type=str, default=None,
                    choices=["MakeBenchmark", "PreProcess", "Train", "ValidateGeneration", "ValidateInference",
                             "Infer"])
parser.add_argument('--specific-file', help='Run for a specific file_name', type=str, default=None)
parser.add_argument('--specific-val-ind', help='Run for a specific indices when doing validation', type=str,
                    default=None)
parser.add_argument('--specific-scan-ind', help='Run for a specific indices when doing scans', type=str, default=None)
parser.add_argument('--scan-points-per-job', help='Number of scan points in a single job', type=int, default=1)
parser.add_argument('--disable-tqdm', help='Disable tqdm print out when training.', action='store_true')
parser.add_argument('--sge-queue', help='Queue for SGE submission', type=str, default="hep.q")
parser.add_argument('--sub-step', help='Sub-step to run for ValidateInference or Infer steps', type=str,
                    default="InitialFit", choices=["InitialFit", "Scan", "Collect", "Plot"])
parser.add_argument('--lower-validation-stats', help='Lowers the validation stats, so code will run faster.', type=int,
                    default=None)
args = parser.parse_args()

if args.cfg is None and args.benchmark is None:
    raise ValueError("The --cfg or --benchmark is required.")
if args.step is None:
    raise ValueError("The --step is required.")

if args.cfg is not None:
    if "Benchmark" in args.cfg:
        args.benchmark = args.cfg.split("Benchmark_")[1].split(".yaml")[0]

if args.benchmark:
    from benchmarks import Benchmarks

    benchmark = Benchmarks(name=args.benchmark)
    if args.step == "MakeBenchmark":
        print("- Making benchmark inputs")

        if args.submit is not None:
            # Submit to batch
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i])}"
            options = {"submit_to": args.submit, "cmds": [cmd],
                       "job_name": f"jobs/innfer_{args.step.lower()}_{args.benchmark}.sh", "sge_queue": args.sge_queue}
            from batch import Batch

            sub = Batch(options=options)
            sub.Run()
            exit()
        else:
            # Make dataset and config
            benchmark.MakeDataset()
            benchmark.MakeConfig()

    args.cfg = f"configs/run/Benchmark_{args.benchmark}.yaml"

with open(args.cfg, 'r') as yaml_file:
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

# TODO check how to get this in a format for sweep
with open(args.architecture, 'r') as yaml_file:
    architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)
    # sweep_id = wandb.sweep(sweep=architecture, project="sweep") # comment out if not sweeping
    run = wandb.init(project='INNFER', config=architecture)

if not os.path.isdir(f"data/{cfg['name']}"): os.system(f"mkdir data/{cfg['name']}")
if not os.path.isdir(f"models/{cfg['name']}"): os.system(f"mkdir models/{cfg['name']}")
if not os.path.isdir(f"plots/{cfg['name']}"): os.system(f"mkdir plots/{cfg['name']}")
for file_name, _ in cfg["files"].items():
    if not os.path.isdir(f"data/{cfg['name']}/{file_name}"): os.system(f"mkdir data/{cfg['name']}/{file_name}")
    if not os.path.isdir(f"plots/{cfg['name']}/{file_name}"): os.system(f"mkdir plots/{cfg['name']}/{file_name}")
    for plot_type in ["preprocess", "train", "validategeneration", "validateinference"]:
        if not os.path.isdir(f"plots/{cfg['name']}/{file_name}/{plot_type}"): os.system(
            f"mkdir plots/{cfg['name']}/{file_name}/{plot_type}")
    for data_type in ["preprocess", "validateinference"]:
        if not os.path.isdir(f"data/{cfg['name']}/{file_name}/{data_type}"): os.system(
            f"mkdir data/{cfg['name']}/{file_name}/{data_type}")

if cfg["preprocess"]["standardise"] == "all":
    cfg["preprocess"]["standardise"] = cfg["variables"] + cfg["pois"] + cfg["nuisances"]

networks = {}
parameters = {}
pp = {}

for file_name, parquet_name in cfg["files"].items():

    # Skip if condition not met
    if args.specific_file != None and args.specific_file != file_name and args.specific_file != "combined": continue

    # Submit to batch
    if args.step in ["PreProcess", "Train"] and args.submit is not None:
        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i])} --specific-file={file_name} --disable-tqdm"
        options = {"submit_to": args.submit, "cmds": [cmd],
                   "job_name": f"jobs/{cfg['name']}/{args.step.lower()}/innfer_{args.step.lower()}_{cfg['name']}_{file_name}.sh",
                   "sge_queue": args.sge_queue}
        from batch import Batch

        sub = Batch(options=options)
        sub.Run()
        continue

    ### PreProcess Data ###
    if args.step == "PreProcess":
        print("- Preprocessing data")

        # PreProcess the dataset
        from preprocess import PreProcess

        pp[file_name] = PreProcess(parquet_name, cfg["variables"], cfg["pois"] + cfg["nuisances"],
                                   options=cfg["preprocess"])
        pp[file_name].output_dir = f"data/{cfg['name']}/{file_name}/{args.step.lower()}"
        pp[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}/{args.step.lower()}"
        pp[file_name].Run()

        # Run plots varying the pois across the variables
        for info in GetPOILoop(cfg, pp[file_name].parameters):
            pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="train",
                                extra_name=f'{info["extra_name"]}_train')
            pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="test",
                                extra_name=f'{info["extra_name"]}_test')
            pp[file_name].PlotX(info["poi"], freeze=info["freeze"], dataset="val",
                                extra_name=f'{info["extra_name"]}_val')

        # Run plots varying the nuisances across the variables for each unique value of the pois
        for info in GetNuisanceLoop(cfg, pp[file_name].parameters):
            pp[file_name].PlotX(info["nuisance"], freeze=info["freeze"], dataset="train",
                                extra_name=f'{info["extra_name"]}_train')

        # Run plots of the distribution of the context features
        pp[file_name].PlotY(dataset="train")

    ### Training, validation and inference ####
    if args.step in ["Train", "ValidateGeneration", "ValidateInference", "Infer"]:

        with open(f"data/{cfg['name']}/{file_name}/preprocess/parameters.yaml", 'r') as yaml_file:
            parameters[file_name] = yaml.load(yaml_file, Loader=yaml.FullLoader)

        if args.submit is None:

            if args.step != "Train":
                with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'r') as yaml_file:
                    architecture = yaml.load(yaml_file, Loader=yaml.FullLoader)

            print("- Building network")
            from network import Network

            networks[file_name] = Network(
                f"data/{cfg['name']}/{file_name}/preprocess/X_train.parquet",
                f"data/{cfg['name']}/{file_name}/preprocess/Y_train.parquet",
                f"data/{cfg['name']}/{file_name}/preprocess/wt_train.parquet",
                f"data/{cfg['name']}/{file_name}/preprocess/X_test.parquet",
                f"data/{cfg['name']}/{file_name}/preprocess/Y_test.parquet",
                f"data/{cfg['name']}/{file_name}/preprocess/wt_test.parquet",
                options=architecture)

            networks[file_name].plot_dir = f"plots/{cfg['name']}/{file_name}/{args.step.lower()}"

            ### Train or load networks ###
            if args.step == "Train":
                print("- Training model")
                networks[file_name].BuildModel()
                networks[file_name].disable_tqdm = args.disable_tqdm
                networks[file_name].BuildTrainer()
                networks[file_name].Train()
                wandb.finish()
                print("saving")
                networks[file_name].Save(name=f"models/{cfg['name']}/{file_name}.h5")

                with open(f"models/{cfg['name']}/{file_name}_architecture.yaml", 'w') as file:
                    yaml.dump(architecture, file)
            else:
                print("- Loading model")
                networks[file_name].Load(name=f"models/{cfg['name']}/{file_name}.h5")
                networks[file_name].data_parameters = parameters[file_name]

        if args.specific_file == "combined": continue

        ### Do validation of trained model ###
        if args.step in ["ValidateGeneration", "ValidateInference"]:

            print("- Performing validation")

            if args.submit is None:
                from validation import Validation

                val = Validation(
                    networks[file_name],
                    options={
                        "data_parameters": parameters[file_name],
                        "pois": cfg["pois"],
                        "nuisances": cfg["nuisances"],
                        "data_dir": f"data/{cfg['name']}/{file_name}/preprocess",
                        "out_dir": f"data/{cfg['name']}/{file_name}/{args.step.lower()}",
                        "plot_dir": f"plots/{cfg['name']}/{file_name}/{args.step.lower()}",
                        "model_name": file_name,
                        "lower_validation_stats": args.lower_validation_stats if args.step == "ValidateInference" else None,
                    }
                )
            # if len(parameters[file_name]["X_columns"]) == 1 and len(parameters[file_name]["Y_columns"]) > 0 and args.step == "ValidateGeneration" and not args.submit:
            #  val.DrawProbability(parameters[file_name]["unique_Y_values"][parameters[file_name]["Y_columns"][0]], n_bins=40)

            for ind, info in enumerate(GetValidateLoop(cfg, parameters[file_name])):

                if args.specific_val_ind != None and args.specific_val_ind != str(ind): continue

                print(f" - Columns: {info['columns']}")
                print(f" - Values: {info['row']}")

                if args.step == "ValidateGeneration":

                    print("- Running generation validation")

                    # Submit to batch
                    if args.submit is not None:
                        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind}"
                        options = {"submit_to": args.submit, "cmds": [cmd],
                                   "job_name": f"jobs/{cfg['name']}/{args.step.lower()}/innfer_{args.step.lower()}_{cfg['name']}_{file_name}_{ind}.sh",
                                   "sge_queue": args.sge_queue}
                        from batch import Batch

                        sub = Batch(options=options)
                        sub.Run()
                        continue

                    # Plot synthetic vs simulated comparison
                    # networks[file_name].ProbabilityIntegral(np.array([info["row"]]), y_columns=info["columns"], n_integral_bins=int((10**6)**(1/len(parameters[file_name]["X_columns"]))))
                    val.PlotGeneration(info["row"], columns=info["columns"])
                    if len(parameters[file_name]["X_columns"]) > 1:
                        val.Plot2DUnrolledGeneration(info["row"], columns=info["columns"])
                        val.PlotCorrelationMatrix(info["row"], columns=info["columns"])

                elif args.step == "ValidateInference":

                    if len(info["row"]) == 0: continue

                    print("- Running validation inference")

                    # Submit to batch
                    if args.submit is not None and not args.sub_step == "Scan":
                        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind}"
                        options = {"submit_to": args.submit, "cmds": [cmd],
                                   "job_name": f"jobs/{cfg['name']}/{args.step.lower()}/{args.sub_step.lower()}/innfer_{args.step.lower()}_{args.sub_step.lower()}_{cfg['name']}_{file_name}_{ind}.sh",
                                   "sge_queue": args.sge_queue}
                        from batch import Batch

                        sub = Batch(options=options)
                        sub.Run()
                        continue

                    # Build an unbinned likelihood for a single model with no constraints
                    if args.submit is None:

                        val.BuildLikelihood()

                        if args.sub_step == "InitialFit":

                            print("- Running initial fit")

                            # Get the best fit
                            val.GetAndDumpBestFit(info["row"], info["initial_best_fit_guess"], ind=ind)
                            # Get scan ranges
                            for col in info["columns"]:
                                val.GetAndDumpScanRanges(info["row"], col, ind=ind)

                    if args.sub_step == "Scan":

                        print("- Running scans")
                        # Load best fit
                        with open(f"data/{cfg['name']}/{file_name}/{args.step.lower()}/best_fit_{ind}.yaml",
                                  'r') as yaml_file:
                            best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                        if args.submit is None:
                            val.lkld.best_fit = np.array(best_fit_info["best_fit"])
                            val.lkld.best_fit_nll = best_fit_info["best_fit_nll"]

                        # Loop through Y values to get a scan for each Y value with the others profiled
                        job_ind = 0
                        for col in info["columns"]:

                            # Load scan ranges
                            with open(
                                    f"data/{cfg['name']}/{file_name}/{args.step.lower()}/scan_values_{col}_{ind}.yaml",
                                    'r') as yaml_file:
                                scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                            # Run scans
                            for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):

                                if not (args.specific_scan_ind != None and args.specific_scan_ind != str(job_ind)):
                                    if args.submit is None:
                                        # val.GetAndDumpNLL(info["row"], col, scan_value, ind1=ind, ind2=point_ind)
                                        val.GetAndDumpScan(info["row"], col, scan_value, ind1=ind, ind2=point_ind)

                                if ((point_ind + 1) % args.scan_points_per_job == 0) or (
                                        point_ind == len(scan_values_info["scan_values"]) - 1):
                                    # Submit to batch
                                    if args.submit is not None:
                                        cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--specific-scan-ind' not in i])} --specific-file={file_name} --specific-val-ind={ind} --specific-scan-ind={job_ind}"
                                        options = {"submit_to": args.submit, "cmds": [cmd],
                                                   "job_name": f"jobs/{cfg['name']}/{args.step.lower()}/{args.sub_step.lower()}/innfer_{args.step.lower()}_{args.sub_step.lower()}_{cfg['name']}_{file_name}_{ind}_{job_ind}.sh",
                                                   "sge_queue": args.sge_queue}
                                        from batch import Batch

                                        sub = Batch(options=options)
                                        sub.Run()
                                    job_ind += 1

                    if args.sub_step == "Collect":

                        print("- Collecting scans")

                        for col in info["columns"]:
                            # Load scan ranges
                            with open(
                                    f"data/{cfg['name']}/{file_name}/{args.step.lower()}/scan_values_{col}_{ind}.yaml",
                                    'r') as yaml_file:
                                scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                            # Load scan results
                            scan_results = {"nlls": [], "scan_values": []}
                            for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):
                                with open(
                                        f"data/{cfg['name']}/{file_name}/{args.step.lower()}/scan_results_{col}_{ind}_{point_ind}.yaml",
                                        'r') as yaml_file:
                                    scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                                if point_ind == 0:
                                    scan_results["row"] = scan_results_info["row"]
                                    scan_results["columns"] = scan_results_info["columns"]
                                    scan_results["varied_column"] = scan_results_info["varied_column"]
                                if None in scan_results_info["nlls"]: continue
                                scan_results["scan_values"] += scan_results_info["scan_values"]
                                scan_results["nlls"] += scan_results_info["nlls"]

                            # Recheck minimum
                            min_nll = min(scan_results["nlls"])
                            scan_results["nlls"] = [i - min_nll for i in scan_results["nlls"]]
                            # Get crossings
                            scan_results["crossings"] = val.lkld.FindCrossings(scan_results["scan_values"],
                                                                               scan_results["nlls"], crossings=[1, 2])
                            # Dump to yaml
                            filename = f"data/{cfg['name']}/{file_name}/{args.step.lower()}/scan_results_{col}_{ind}.yaml"
                            print(f">> Created {filename}")
                            with open(filename, 'w') as yaml_file:
                                yaml.dump(scan_results, yaml_file, default_flow_style=False)
                            # Remove per point yaml files
                            for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):
                                os.system(
                                    f"rm data/{cfg['name']}/{file_name}/{args.step.lower()}/scan_results_{col}_{ind}_{point_ind}.yaml")

                    if args.sub_step == "Plot":

                        print("- Plotting scans and comparisons")

                        with open(f"data/{cfg['name']}/{file_name}/{args.step.lower()}/best_fit_{ind}.yaml",
                                  'r') as yaml_file:
                            best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                        for col in info["columns"]:
                            print(f"data/{cfg['name']}/{file_name}/{args.step.lower()}/scan_results_{col}_{ind}.yaml")
                            with open(
                                    f"data/{cfg['name']}/{file_name}/{args.step.lower()}/scan_results_{col}_{ind}.yaml",
                                    'r') as yaml_file:
                                scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                            if args.benchmark is not None:
                                pdf = benchmark.GetPDF(file_name)
                            else:
                                pdf = None

                            val.PlotLikelihood(scan_results_info["scan_values"], scan_results_info["nlls"],
                                               scan_results_info["row"], col, scan_results_info["crossings"],
                                               best_fit_info["best_fit"], true_pdf=pdf)
                        val.PlotComparisons(best_fit_info["row"], best_fit_info["best_fit"])

# do combined inference validation
if args.step == "ValidateInference" and (args.specific_file == None or args.specific_file == "combined") and len(
        cfg["files"].keys()) > 1:

    if args.submit is None:
        from likelihood import Likelihood

        lkld = Likelihood(
            {
                "pdfs": {k: networks[k] for k in cfg["files"]},
                "yields": {k: MakeYieldFunction(cfg["pois"], cfg["nuisances"], parameters[k]) for k in cfg["files"]}
            },
            type="unbinned_extended",
            data_parameters={k: parameters[k] for k in cfg["files"]},
            parameters=cfg["inference"],
        )

    # Loop through validation iterations
    for ind, info in enumerate(GetCombinedValidateLoop(cfg, parameters)):

        if args.specific_val_ind != None and args.specific_val_ind != str(ind): continue

        print(f" - Columns: {info['columns']}")
        print(f" - Values: {info['row']}")

        if args.submit is not None and args.sub_step != "Scan":
            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i])} --specific-file=combined --specific-val-ind={ind}"
            options = {"submit_to": args.submit, "cmds": [cmd],
                       "job_name": f"jobs/{cfg['name']}/{args.step.lower()}/{args.sub_step.lower()}/innfer_{args.step.lower()}_{args.sub_step.lower()}_{cfg['name']}_combined_{ind}.sh",
                       "sge_queue": args.sge_queue}
            from batch import Batch

            sub = Batch(options=options)
            sub.Run()
            continue

        # Make toy dataset
        first_loop = True
        for file_name, parquet_name in cfg["files"].items():

            # Set up PreProcess
            from preprocess import PreProcess

            pp[file_name] = PreProcess(parquet_name, cfg["variables"], cfg["pois"] + cfg["nuisances"],
                                       options=cfg["preprocess"])
            pp[file_name].output_dir = f"data/{cfg['name']}/{file_name}/preprocess"
            pp[file_name].parameters = parameters[file_name]

            # Get dataset
            X, Y, wt = pp[file_name].LoadSplitData(dataset="val", get=["X", "Y", "wt"], use_nominal_wt=False)
            X = X.to_numpy()
            Y = Y.to_numpy()
            wt = wt.to_numpy()
            row = [info["row"][info["columns"].index(y)] for y in parameters[file_name]["Y_columns"]]

            # Cut dataset
            if Y.shape[1] > 0:
                matching_rows = np.all(np.isclose(Y, np.array(row), rtol=1e-6, atol=1e-6), axis=1)
                X = X[matching_rows]
                wt = wt[matching_rows]

            # Scale wt to toy value
            if "yield" in parameters[file_name].keys():
                if "all" in parameters[file_name]["yield"].keys():
                    sum_wt = parameters[file_name]["yield"]["all"]
                else:
                    sum_wt = parameters[file_name]["yield"][GetYName(row, purpose="file")]
                old_sum_wt = np.sum(wt)
                wt *= sum_wt / old_sum_wt

            # Scale wt by the rate parameter value
            if "mu_" + file_name in info["columns"]:
                rp = info["row"][info["columns"].index("mu_" + file_name)]
                if rp == 0.0: continue
                wt *= rp

            if args.lower_validation_stats is not None:
                if len(X) > args.lower_validation_stats:
                    random_indices = np.random.choice(X.shape[0], args.lower_validation_stats, replace=False)
                    X = X[random_indices, :]
                    sum_wt = np.sum(wt)
                    wt = wt[random_indices, :]
                    new_sum_wt = np.sum(wt)
                    wt *= sum_wt / new_sum_wt

            # Combine datasets
            if first_loop:
                total_X = copy.deepcopy(X)
                total_wt = copy.deepcopy(wt)
                first_loop = False
            else:
                total_X = np.vstack((total_X, X))
                total_wt = np.vstack((total_wt, wt))

        indices = np.arange(total_X.shape[0])
        np.random.shuffle(indices)
        total_X = total_X[indices]
        total_wt = total_wt[indices]

        if not os.path.isdir(f"data/{cfg['name']}/combined"): os.system(f"mkdir data/{cfg['name']}/combined")
        out_dir = f"data/{cfg['name']}/combined/{args.step.lower()}/"
        if not os.path.isdir(out_dir): os.system(f"mkdir {out_dir}")

        if args.sub_step == "InitialFit":
            print("- Running initial fit")

            # loop = [166.5,169.5,171.5,172.5,173.5,175.5,178.5]
            # for i in loop:
            #  lkld.Run(total_X, np.array([0.5,i]), wts=total_wt, return_ln=True)
            # exit()

            lkld.GetAndWriteBestFitToYaml(total_X, info["row"], info["initial_best_fit_guess"], wt=total_wt,
                                          filename=f"{out_dir}/best_fit_{ind}.yaml")
            for col in info["columns"]:
                lkld.GetAndWriteScanRangesToYaml(total_X, info["row"], col, wt=total_wt,
                                                 filename=f"{out_dir}/scan_ranges_{col}_{ind}.yaml")

        elif args.sub_step == "Scan":
            print("- Running scans")

            # Load best fit
            with open(f"{out_dir}/best_fit_{ind}.yaml", 'r') as yaml_file:
                best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

            if args.submit is None:
                lkld.best_fit = np.array(best_fit_info["best_fit"])
                lkld.best_fit_nll = best_fit_info["best_fit_nll"]

            # Loop through Y values to get a scan for each Y value with the others profiled
            job_ind = 0
            for col in info["columns"]:

                # Load scan ranges
                with open(f"{out_dir}/scan_ranges_{col}_{ind}.yaml", 'r') as yaml_file:
                    scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                # Run scans
                for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):

                    if not (args.specific_scan_ind != None and args.specific_scan_ind != str(job_ind)):
                        if args.submit is None:
                            # lkld.GetAndWriteNLLToYaml(total_X, info["row"], col, scan_value, wt=total_wt, filename=f"{out_dir}/scan_results_{col}_{ind}_{point_ind}.yaml")
                            lkld.GetAndWriteScanToYaml(total_X, info["row"], col, scan_value, wt=total_wt,
                                                       filename=f"{out_dir}/scan_results_{col}_{ind}_{point_ind}.yaml")

                    if ((point_ind + 1) % args.scan_points_per_job == 0) or (
                            point_ind == len(scan_values_info["scan_values"]) - 1):
                        # Submit to batch
                        if args.submit is not None:
                            cmd = f"python3 {' '.join([i for i in sys.argv if '--submit' not in i and '--specific-file' not in i and '--specific-val-ind' not in i and '--specific-scan-ind' not in i])} --specific-file=combined --specific-val-ind={ind} --specific-scan-ind={job_ind}"
                            options = {"submit_to": args.submit, "cmds": [cmd],
                                       "job_name": f"jobs/{cfg['name']}/{args.step.lower()}/{args.sub_step.lower()}/innfer_{args.step.lower()}_{args.sub_step.lower()}_{cfg['name']}_combined_{ind}_{job_ind}.sh",
                                       "sge_queue": args.sge_queue}
                            from batch import Batch

                            sub = Batch(options=options)
                            sub.Run()
                        job_ind += 1

        elif args.sub_step == "Collect":
            print("- Collecting scans")

            for col in info["columns"]:
                # Load scan ranges
                with open(f"{out_dir}/scan_ranges_{col}_{ind}.yaml", 'r') as yaml_file:
                    scan_values_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                # Load scan results
                scan_results = {"nlls": [], "scan_values": []}
                for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):
                    with open(f"{out_dir}/scan_results_{col}_{ind}_{point_ind}.yaml", 'r') as yaml_file:
                        scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    if point_ind == 0:
                        scan_results["row"] = scan_results_info["row"]
                        scan_results["columns"] = scan_results_info["columns"]
                        scan_results["varied_column"] = scan_results_info["varied_column"]
                    if any(np.isnan(x) for x in scan_results_info["nlls"]): continue
                    scan_results["scan_values"] += scan_results_info["scan_values"]
                    scan_results["nlls"] += scan_results_info["nlls"]

                # Recheck minimum
                min_nll = min(scan_results["nlls"])
                scan_results["nlls"] = [i - min_nll for i in scan_results["nlls"]]
                # Get crossings
                scan_results["crossings"] = lkld.FindCrossings(scan_results["scan_values"], scan_results["nlls"],
                                                               crossings=[1, 2])
                # Dump to yaml
                filename = f"{out_dir}/scan_results_{col}_{ind}.yaml"
                print(f">> Created {filename}")
                with open(filename, 'w') as yaml_file:
                    yaml.dump(scan_results, yaml_file, default_flow_style=False)
                # Remove per point yaml files
                for point_ind, scan_value in enumerate(scan_values_info["scan_values"]):
                    os.system(f"rm {out_dir}/scan_results_{col}_{ind}_{point_ind}.yaml")

        elif args.sub_step == "Plot":

            with open(f"{out_dir}/best_fit_{ind}.yaml", 'r') as yaml_file:
                best_fit_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

            if not os.path.exists(f"plots/{cfg['name']}/combined"): os.system(f"mkdir plots/{cfg['name']}/combined")
            plot_dir = f"plots/{cfg['name']}/combined/{args.step.lower()}"
            if not os.path.exists(plot_dir): os.system(f"mkdir {plot_dir}")

            print("- Plotting scans")
            for col in info["columns"][::-1]:
                with open(f"{out_dir}/scan_results_{col}_{ind}.yaml", 'r') as yaml_file:
                    scan_results_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

                file_extra_name = GetYName(scan_results_info["row"], purpose="file")
                plot_extra_name = GetYName(scan_results_info["row"], purpose="plot")

                other_lkld = {}

                eff_events = float((np.sum(total_wt.flatten()) ** 2) / np.sum(total_wt.flatten() ** 2))
                sum_weight = float(np.sum(total_wt.flatten()))
                if not eff_events == sum_weight:
                    other_lkld[r"Inferred N=$N_{eff}$"] = (eff_events / sum_weight) * np.array(
                        scan_results_info["nlls"])

                if args.benchmark is not None:

                    def Probability(X, Y, y_columns=None, k=None, **kwargs):
                        Y = np.array(Y)
                        if y_columns is not None:
                            column_indices = [y_columns.index(col) for col in parameters[k]["Y_columns"]]
                            Y = Y[:, column_indices]
                        if len(Y) == 1: Y = np.tile(Y, (len(X), 1))
                        true_pdf = benchmark.GetPDF(k)
                        prob = np.zeros(len(X))
                        for i in range(len(X)):
                            prob[i] = true_pdf(X[i], Y[i])
                        return np.log(prob)


                    for k in lkld.models["pdfs"].keys():
                        lkld.models["pdfs"][k].Probability = partial(Probability, k=k)

                    nlls = []
                    for x_val in scan_results_info["scan_values"]:
                        test_row = copy.deepcopy(best_fit_info["row"])
                        test_row[best_fit_info["columns"].index(col)] = x_val
                        nlls.append(-2 * lkld.Run(total_X, test_row, wts=total_wt, return_ln=True))

                    min_nll = min(nlls)
                    nlls = [nll - min_nll for nll in nlls]
                    other_lkld["True"] = nlls

                plot_likelihood(
                    scan_results_info["scan_values"],
                    scan_results_info["nlls"],
                    scan_results_info["crossings"],
                    name=f"{plot_dir}/likelihood_{col}_y_{file_extra_name}",
                    xlabel=col,
                    true_value=scan_results_info["row"][scan_results_info["columns"].index(col)],
                    title_right=f"y={plot_extra_name}",
                    cap_at=9,
                    label=f"Inferred N={int(round(sum_weight))}",
                    other_lklds=other_lkld,
                )

            print("- Plotting distributions")

            total_events_per_file = 10 ** 6
            synth_datasets = {}
            synth_weights = {}
            for key, pdf in lkld.models["pdfs"].items():
                synth_weights[key] = np.ones(total_events_per_file) / total_events_per_file
                if "yield" in parameters[key].keys():
                    func = MakeYieldFunction(cfg["pois"], cfg["nuisances"], parameters[key])
                    scale = func(row)
                    synth_weights[key] *= scale
                else:
                    synth_weights[key] *= np.sum(total_wt)
                if "mu_" + key in lkld.Y_columns:
                    synth_weights[key] *= best_fit_info["best_fit"][best_fit_info["columns"].index("mu_" + key)]
                synth_datasets[key] = pdf.Sample(best_fit_info["best_fit"], columns=best_fit_info["columns"],
                                                 n_events=total_events_per_file)

            for ind, col in enumerate(cfg["variables"]):
                n_bins = 40
                ignore_quantile = 0.01
                trimmed_X = total_X[:, ind]
                lower_value = np.quantile(trimmed_X, ignore_quantile)
                upper_value = np.quantile(trimmed_X, 1 - ignore_quantile)
                trimmed_indices = ((trimmed_X >= lower_value) & (trimmed_X <= upper_value))
                trimmed_X = trimmed_X[trimmed_indices]
                trimmed_wt = total_wt[trimmed_indices].flatten()

                data_hist, bins = np.histogram(trimmed_X, weights=trimmed_wt, bins=n_bins)
                data_err_hist, _ = np.histogram(trimmed_X, weights=trimmed_wt ** 2, bins=bins)
                data_err_hist = np.sqrt(data_err_hist)
                synth_hists = {}
                total_synth_hist_squared = np.zeros(n_bins)
                for k, v in synth_datasets.items():
                    synth_hist, _ = np.histogram(v[:, ind], bins=bins, weights=synth_weights[k])
                    total_synth_hist_squared += np.histogram(v[:, ind], bins=bins, weights=synth_weights[k] ** 2)[0]
                    # synth_hists[k] = (sum_X_weights/total_events)*synth_hist.astype(float)
                    synth_hists[k] = synth_hist.astype(float)
                total_synth_error = np.sqrt(total_synth_hist_squared)
                # total_synth_error *= (sum_X_weights/total_events)

                file_extra_name = GetYName(best_fit_info["row"], purpose="file")
                plot_extra_name = GetYName(best_fit_info["row"], purpose="plot")
                bf_plot_extra_name = GetYName(best_fit_info["best_fit"], purpose="plot")

                plot_stacked_histogram_with_ratio(
                    data_hist,
                    synth_hists,
                    bins,
                    data_name='Data',
                    xlabel=col,
                    name=f"{plot_dir}/comparison_{col}_y_{file_extra_name}",
                    data_errors=data_err_hist,
                    stack_hist_errors=total_synth_error,
                    title_right=f"y={plot_extra_name}",
                    use_stat_err=False,
                    axis_text=f"Best Fit y={bf_plot_extra_name}"
                )

print("- Finished running without error")
