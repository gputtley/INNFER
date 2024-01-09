import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--plot-dir', help= 'Directory of Plots',  default="plots/Benchmark_Gaussian")
parser.add_argument('--max-per-slide', help= 'Max plots per slide', type=int, default=6)
parser.add_argument('--max-width', help= 'Max width for each plot', type=float, default=0.6)
args = parser.parse_args()


name = "combined_plots"
if os.path.isfile(f"{args.plot_dir}/{name}.pdf"): 
  os.system(f"rm {args.plot_dir}/{name}.pdf")

def GetPDFs(directory, max_per_slide=6):
  # Gets PDFs in each folder
  pdf_files = {}
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(".pdf"):

        key = " ".join([item for item in root.split(directory)[1].split("/") if item != ""])

        if key not in pdf_files.keys():
          pdf_files[key] = [f"{root}/{file}"]
        else:
          pdf_files[key].append(f"{root}/{file}")

  # Split by different namings
  separate_frames = {}
  inds = {}
  for k, v in pdf_files.items():
    
    if len(v) <= 2:
      separate_frames[k] = sorted(v)
    else:

      for i in sorted(v):

        initial_name = f"{k} {i.split('/')[-1].split('.pdf')[0].split('_')[0]}"

        if initial_name not in inds.keys():
          inds[initial_name] = 1

        name = f"{initial_name} - {inds[initial_name]}" 

        if name in separate_frames.keys():
          if len(separate_frames[name]) >= max_per_slide: 
            inds[initial_name] += 1
            name = f"{initial_name} - {inds[initial_name]}"

        if name not in separate_frames.keys():
          separate_frames[name] = [i]
        else:
          separate_frames[name].append(i)    

  return separate_frames

# latex code
LB = "{"
RB = "}"
BS = "\\"

beamer = [
  f"{BS}documentclass{LB}beamer{RB}",
  f"{BS}usepackage{LB}graphicx{RB}",
  f"{BS}usepackage{LB}subcaption{RB}",
  f"{BS}setbeamersize{LB}text margin left=1mm, text margin right=1mm{RB}"
  f"{BS}begin{LB}document{RB}"
]

for k, frame_plots in GetPDFs(args.plot_dir, max_per_slide=args.max_per_slide).items():
  beamer.append(f"  {BS}begin{LB}frame{RB}{LB}{RB}")
  beamer.append(f"    {BS}frametitle{LB}{k}{RB}")
  beamer.append(f"    {BS}begin{LB}figure{RB}")
  beamer.append(f"      {BS}centering")
  width = min(1.0/min(len(frame_plots),np.ceil(args.max_per_slide)/2.0),args.max_width)


  for plot in frame_plots:
    beamer.append(f"       {BS}begin{LB}subfigure{RB}{LB}{width}{BS}textwidth{RB}")
    beamer.append(f"          {BS}includegraphics[width={BS}linewidth]{LB}{plot}{RB}")
    if plot != frame_plots[-1]:
      beamer.append(f"       {BS}end{LB}subfigure{RB}%")
      beamer.append(f"       {BS}hfill")
    else:
      beamer.append(f"       {BS}end{LB}subfigure{RB}")
  beamer.append(f"    {BS}end{LB}figure{RB}")
  beamer.append(f"  {BS}end{LB}frame{RB}")
beamer.append(f"{BS}end{LB}document{RB}")


def CreateFile(name,cmd_list):
  if os.path.exists(name): os.system('rm %(name)s' % vars())
  for cmd in cmd_list:
    os.system(f'echo "{cmd}" >> {name}')

CreateFile(f"{args.plot_dir}/{name}.tex",beamer)

os.system(f"pdflatex -interaction=nonstopmode -output-directory={args.plot_dir} {args.plot_dir}/{name}.tex > /dev/null 2>&1")
if os.path.isfile(f"{args.plot_dir}/{name}.tex"): os.system(f"rm {args.plot_dir}/{name}.tex")
if os.path.isfile(f"{args.plot_dir}/{name}.aux"): os.system(f"rm {args.plot_dir}/{name}.aux")
if os.path.isfile(f"{args.plot_dir}/{name}.nav"): os.system(f"rm {args.plot_dir}/{name}.nav")
if os.path.isfile(f"{args.plot_dir}/{name}.log"): os.system(f"rm {args.plot_dir}/{name}.log")
if os.path.isfile(f"{args.plot_dir}/{name}.out"): os.system(f"rm {args.plot_dir}/{name}.out")
if os.path.isfile(f"{args.plot_dir}/{name}.snm"): os.system(f"rm {args.plot_dir}/{name}.snm")
if os.path.isfile(f"{args.plot_dir}/{name}.toc"): os.system(f"rm {args.plot_dir}/{name}.toc")
