import math
import numbers
import os
import re
import shutil
import subprocess
import tempfile

from useful_functions import Translate


class DensityMetricsTable():
  """Present saved density metrics without changing their numerical definitions."""

  def __init__(self, metrics, metadata=None, hypotheses=None):
    self.metrics = metrics
    self.metadata = metadata or {}
    self.hypotheses = hypotheses or {}


  def _Escape(self, value):
    replacements = {
      "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
      "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}",
      "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


  def _Label(self, value):
    value = str(value)
    translated = Translate(value)
    return translated if translated != value else self._Escape(value)


  def _Number(self, value, precision=5):
    if value is None:
      return r"\textemdash"
    if isinstance(value, bool):
      return str(value)
    if not isinstance(value, numbers.Real):
      return self._Escape(value)
    if math.isnan(value):
      return r"$\mathrm{NaN}$"
    if math.isinf(value):
      return r"$-\infty$" if value < 0 else r"$\infty$"
    if isinstance(value, numbers.Integral):
      return str(value)
    formatted = f"{value:.{precision}g}"
    if "e" in formatted:
      mantissa, exponent = formatted.split("e")
      return rf"${mantissa}\times 10^{{{int(exponent)}}}$"
    return formatted


  def _Table(self, title, headers, rows, widths, numeric_columns=(), note=None, rules_before=()):
    if not rows:
      return ""
    columns = "".join(
      rf">{{\raggedleft\arraybackslash}}p{{{width}\linewidth}}" if ind in numeric_columns
      else rf">{{\raggedright\arraybackslash}}p{{{width}\linewidth}}"
      for ind, width in enumerate(widths)
    )
    header = " & ".join(r"\textbf{" + label + "}" for label in headers) + r" \\"
    lines = [r"\par\smallskip{\bfseries " + title + r"}\par"]
    if note:
      lines.append(note + r"\par")
    lines += [
      r"\begin{longtable}{@{}" + columns + "@{}}",
      r"\toprule", header, r"\midrule", r"\endfirsthead",
      r"\toprule", header, r"\midrule", r"\endhead",
      r"\midrule", rf"\multicolumn{{{len(headers)}}}{{r}}{{\small Continued on next page}} \\",
      r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    for ind, row in enumerate(rows):
      if ind in rules_before:
        lines.append(r"\specialrule{0.3pt}{0.3pt}{0.3pt}")
      lines.append(" & ".join(row) + r" \\")
    lines += [r"\end{longtable}"]
    return "\n".join(lines)


  def _Flatten(self, value, prefix=""):
    if isinstance(value, dict):
      for key, entry in sorted(value.items(), key=lambda item: str(item[0])):
        yield from self._Flatten(entry, f"{prefix} / {key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
      for ind, entry in enumerate(value):
        yield from self._Flatten(entry, f"{prefix} / {ind}")
    else:
      yield prefix, value


  def _Matrix(self, title, entries, label_heading="Metric"):
    """Pivot hypotheses into columns, splitting wide scans into groups of six."""
    if not entries:
      return ""
    indices = sorted({ind for values in entries.values() for ind in values if ind is not None})
    has_aggregate = any(None in values for values in entries.values())
    columns = indices + ([None] if has_aggregate else [])
    tables = []
    for offset in range(0, len(columns), 6):
      group = columns[offset:offset + 6]
      headers = ["Dataset", label_heading] + [self._HypothesisLabel(ind) for ind in group]
      dataset_order = {dataset: ind for ind, dataset in enumerate(dict.fromkeys(dataset for dataset, _ in entries))}
      selected = [(dataset, label, values) for (dataset, label), values in entries.items()
        if any(ind in values for ind in group)]
      selected.sort(key=lambda entry: dataset_order[entry[0]])
      rows = []
      rules_before = []
      previous_dataset = None
      for dataset, label, values in selected:
        if rows and (dataset != previous_dataset or label in ["Mean over observables", "Sum over observables"]):
          rules_before.append(len(rows))
        rows.append([{"train_inf": "Train", "test_inf": "Test", "val": "Validation", "full": "Full"}.get(dataset, self._Escape(dataset)), label] +
          [values.get(ind, r"\textemdash") for ind in group])
        previous_dataset = dataset
      tables.append(self._Table(title, headers, rows,
        [0.08, 0.23] + [0.63 / len(group)] * len(group), range(2, len(headers)), rules_before=rules_before))
    return "\n".join(tables)


  def _HypothesisLabel(self, ind):
    if ind is None:
      return "All / merged"
    if not self.hypotheses.get(ind):
      return f"Index {ind}"
    labels = []
    for key, value in self.hypotheses[ind].items():
      translated = Translate(str(key), only_val=True)
      label = translated if translated != str(key) else self._Escape(key)
      unit = Translate(str(key), only_unit=True, only_unit_with_space=False)
      labels.append(f"{label} = {self._Number(value, 7)}" + (" " + self._Escape(unit) if unit else ""))
    return r"\newline ".join(labels)


  def BuildLatex(self):
    """Return compact tables with validation hypotheses arranged side by side."""
    sections = []
    consumed = set()
    loss_labels = {
      "loss_train": "Train", "loss_test": "Test",
      "loss_difference": "Test minus train",
      "loss_test_plus_difference": "Test plus difference",
    }
    loss_keys = [key for key in loss_labels if key in self.metrics]
    if loss_keys:
      sections.append(self._Table("Flow loss", [loss_labels[key] for key in loss_keys],
        [[self._Number(self.metrics[key], 7) for key in loss_keys]], [0.94 / len(loss_keys)] * len(loss_keys),
        range(len(loss_keys))))
      consumed.update(loss_keys)

    histogram = {}
    pattern = r"^(chi_squared_per_dof|dof_for_chi_squared|chi_squared|kl_divergence)_(.+)_val_ind_(\d+)$"
    for key, values in sorted(self.metrics.items()):
      match = re.match(pattern, key)
      if match and isinstance(values, dict):
        metric, dataset, ind = match.groups()
        for observable, value in values.items():
          histogram.setdefault((dataset, observable), {}).setdefault(int(ind), {})[metric] = value
        consumed.add(key)
    histogram_rows = {}
    kl_rows = {}
    for (dataset, observable), hypotheses in sorted(histogram.items()):
      label = {"mean": "Mean over observables", "sum": "Sum over observables"}.get(observable) or self._Label(observable)
      for ind, values in hypotheses.items():
        parts = []
        if "chi_squared_per_dof" in values:
          parts.append(self._Number(values["chi_squared_per_dof"], 4))
        if "chi_squared" in values or "dof_for_chi_squared" in values:
          raw = self._Number(values.get("chi_squared"), 4) + "/" + self._Number(values.get("dof_for_chi_squared"))
          parts.append("(" + raw + ")")
        if parts:
          histogram_rows.setdefault((dataset, label), {})[ind] = " ".join(parts)
        if "kl_divergence" in values:
          kl_rows.setdefault((dataset, label), {})[ind] = self._Number(values["kl_divergence"])
    # Place dataset-wide arithmetic aggregates in the same rows as per-hypothesis means/sums.
    for key, value in sorted(self.metrics.items()):
      match = re.fullmatch(r"(chi_squared_per_dof|kl_divergence)_(.+)_(mean|sum)", key)
      if match:
        metric, dataset, aggregate = match.groups()
        target = histogram_rows if metric == "chi_squared_per_dof" else kl_rows
        label = "Mean over observables" if aggregate == "mean" else "Sum over observables"
        target.setdefault((dataset, label), {})[None] = self._Number(value, 4)
        consumed.add(key)
    sections.append(self._Matrix(r"Marginal agreement: $\chi^2/\mathrm{ndf}$ (raw $\chi^2$/ndf)", histogram_rows, "Observable"))
    sections.append(self._Matrix("Marginal agreement: KL divergence", kl_rows, "Observable"))

    multidim_labels = {
      "bdt_signed_accuracy": "Signed accuracy", "bdt_signed_auc": "Signed ROC AUC",
      "bdt_accuracy": "Accuracy", "bdt_auc": "ROC AUC",
      "k_means_chi_squared": r"K-means $\chi^2/\mathrm{ndf}$",
      "wasserstein_sliced": "Sliced Wasserstein",
      "wasserstein_unbinned": "Wasserstein",
    }
    multidim = {}
    for key, value in sorted(self.metrics.items()):
      for prefix, label in multidim_labels.items():
        if not key.startswith(prefix + "_"):
          continue
        suffix = key[len(prefix) + 1:]
        match = re.fullmatch(r"(.+)_val_ind_(\d+)", suffix)
        dataset, ind = (match.group(1), int(match.group(2))) if match else (suffix, None)
        for path, entry in self._Flatten(value):
          row_label = label + (" / " + self._Label(path) if path else "")
          multidim.setdefault((dataset, row_label), {})[ind] = self._Number(entry)
        consumed.add(key)
        break
    sections.append(self._Matrix("Multidimensional comparison", multidim))

    inference_labels = {"best_fit": "Fit", "distance": "Absolute bias", "chi_squared": r"Pull$^2$"}
    inference = {}
    parameters = {match.group(1) for key in self.metrics
      if (match := re.fullmatch(r"inference_(?:best_fit|distance|chi_squared)_.+_val_ind_\d+_(.+)", key))}
    for key, value in sorted(self.metrics.items()):
      match = re.fullmatch(r"inference_(best_fit|distance|chi_squared)_(.+)_val_ind_(\d+)_(.+)", key)
      if match:
        metric, dataset, ind, parameter = match.groups()
        label = (self._Label(parameter) + ": " if len(parameters) > 1 else "") + inference_labels[metric]
        inference.setdefault((dataset, label), {})[int(ind)] = self._Number(value, 7 if metric == "best_fit" else 4)
        consumed.add(key)
    for key, value in sorted(self.metrics.items()):
      if key in consumed:
        continue
      match = re.fullmatch(r"inference_(distance|chi_squared)_(.+)", key)
      if match:
        metric, dataset = match.groups()
        # The runner's dataset-wide aggregates combine all fitted parameters.
        label = inference_labels[metric] + " (all parameters)"
        parameter_labels = [row for row in inference if row[0] == dataset and
          (row[1] == inference_labels[metric] or row[1].endswith(": " + inference_labels[metric]))]
        if len(parameter_labels) == 1:
          row = parameter_labels[0]
        else:
          row = (dataset, label)
        inference.setdefault(row, {})[None] = self._Number(value, 4)
        consumed.add(key)
    inference_title = "Inference closure"
    if len(parameters) == 1:
      inference_title += ": " + self._Label(next(iter(parameters)))
    sections.append(self._Matrix(inference_title, inference))

    remaining = [(self._Escape(path), self._Number(entry)) for key, value in sorted(self.metrics.items())
      if key not in consumed for path, entry in self._Flatten(value, key)]
    # Two metric/value pairs per row keep additional scalar metrics compact.
    extra_rows = []
    for offset in range(0, len(remaining), 2):
      pair = remaining[offset:offset + 2]
      extra_rows.append(list(pair[0]) + (list(pair[1]) if len(pair) == 2 else ["", ""]))
    sections.append(self._Table("Additional metrics", ["Metric", "Value", "Metric", "Value"],
      extra_rows, [0.35, 0.12, 0.35, 0.12], [1, 3]))

    metadata = r"\quad $\vert$ \quad ".join(
      r"\textbf{" + self._Escape(key) + ":} " + self._Label(value)
      for key, value in self.metadata.items() if value is not None and value != ""
    )
    return "\n".join([
      r"\documentclass[10pt,a4paper,landscape]{article}",
      r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
      r"\usepackage[margin=0.8cm]{geometry}", r"\usepackage{booktabs,longtable,array}",
      r"\setlength{\parindent}{0pt}", r"\setlength{\tabcolsep}{3pt}",
      r"\renewcommand{\arraystretch}{1.02}", r"\setlength{\emergencystretch}{3em}",
      r"\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}",
      r"\begin{document}\fontsize{8.5}{9}\selectfont",
      r"{\large\bfseries Density performance metrics}\par\smallskip",
      metadata + r"\par\smallskip",
      *sections,
      r"\par\smallskip\textit{Notes:} Loss excludes L2 and density correction. "
      r"Histogram mean/sum entries aggregate observable scores, not a pooled $\chi^2/\mathrm{ndf}$. "
      r"Closure pull$^2$ uses the uncertainty towards truth. Mean fits average separate dataset fits. "
      r"All/merged denotes dataset-wide aggregates or merged hypotheses. "
      r"Values use 4--7 significant figures; YAML retains full precision. A dash denotes an unavailable value.",
      r"\end{document}", "",
    ])


  def Write(self, output_stem):
    """Write .tex and compile .pdf; keep an actionable log if compilation fails."""
    output_stem = os.path.abspath(os.fspath(output_stem))
    os.makedirs(os.path.dirname(output_stem), exist_ok=True)
    tex_path = output_stem + ".tex"
    with open(tex_path, "w", encoding="utf-8") as tex_file:
      tex_file.write(self.BuildLatex())

    compiler = shutil.which("pdflatex")
    if compiler is None:
      raise RuntimeError(f"Cannot compile {tex_path}: pdflatex is not on PATH. "
        "Install a LaTeX distribution with booktabs, longtable, array and geometry; the YAML and LaTeX have been saved.")

    # Isolate auxiliary files and disable TeX shell execution. Two passes resolve longtable widths.
    with tempfile.TemporaryDirectory(prefix="density_metrics_table_") as directory:
      shutil.copyfile(tex_path, os.path.join(directory, "table.tex"))
      for _ in range(2):
        try:
          result = subprocess.run([compiler, "-no-shell-escape", "-interaction=nonstopmode",
            "-halt-on-error", "table.tex"], cwd=directory, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", timeout=60)
          if result.returncode == 0:
            continue
          diagnostic = result.stdout
        except subprocess.TimeoutExpired as error:
          diagnostic = f"LaTeX compilation timed out after {error.timeout} seconds.\n"
        log_path = output_stem + "_latex.log"
        compiler_log = os.path.join(directory, "table.log")
        if os.path.isfile(compiler_log):
          shutil.copyfile(compiler_log, log_path)
        else:
          with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(diagnostic)
        raise RuntimeError(f"Failed to compile {tex_path}; see {log_path}. The YAML and LaTeX have been saved.")
      shutil.copyfile(os.path.join(directory, "table.pdf"), output_stem + ".pdf")
