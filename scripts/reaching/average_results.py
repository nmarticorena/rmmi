import os
import subprocess
import pandas as pd
from typing import List
from dataclasses import dataclass, field
import tyro
from jinja2 import Environment, FileSystemLoader

jenv = Environment(loader=FileSystemLoader("template/"))


@dataclass
class Config:
    envs: List = field(
        default_factory=lambda: [
            "bookshelf",
            "bookshelf_2",
            "table_free",
            "table",
            "table_cyl_small",
        ]
    )
    exp_name: str = "test_random/"
    file_name: str = "test"


args = tyro.cli(Config)


experiment_name = "logs/" + args.exp_name
file_name = args.file_name
# print(experiment_name)

master_folder = args.envs
data = []
success_data = []
for i in master_folder:
    try:
        data.append(pd.read_csv(f"{experiment_name}/results/{i}.csv", index_col=0))
        success_data.append(
            pd.read_csv(f"{experiment_name}/results/{i}_successfull.csv", index_col=0)
        )
    except FileNotFoundError:
        print(f"File {i} not found")
        continue

final = pd.concat(data).reset_index().groupby("index").mean()
final_success = pd.concat(success_data).reset_index().groupby("index").mean()
# print(final_success)
# print(final_success.rows)
row = final_success.loc["mean eef acc"].to_frame().T
row.index = ["mean eef acc successfull"]
final = pd.concat([final, row], ignore_index=False)
final_std = pd.concat(data).reset_index().groupby("index").std()
max = pd.concat(data).reset_index().groupby("index").max()
final.to_csv(f"{experiment_name}/results/total.csv")
print("saving on ", f"{experiment_name}/results/total.csv")
final.to_latex(f"{experiment_name}/latex_tables/total.tex", float_format="%.3f")
max.to_latex(f"{experiment_name}/latex_tables/max.tex", float_format="%.3f")


final.to_csv(f"{experiment_name}/results/max.csv")
print(final)

job_name = experiment_name.split("/")[-3:]
job_name = "_".join(job_name)
# import pdb; pdb.set_trace()


# job_name = experiment_name.replace('/','_')

# subprocess.run(f"cp template/report.tex {experiment_name}/latex_tables/report.tex", shell = True)
template = jenv.get_template("doc.tex")
# names = [i.replace('_',' ') for i in master_folder]
with open(f"{experiment_name}/latex_tables/report.tex", mode="wt") as f:
    f.write(template.render(tables=master_folder))

subprocess.run(
    f"cd {experiment_name}/latex_tables && pdflatex -jobname {job_name} report.tex",
    shell=True,
)
os.makedirs(f"logs/reports/{file_name}", exist_ok=True)
subprocess.run(
    f"cp {experiment_name}/latex_tables/{job_name}.pdf  logs/reports/{file_name}/",
    shell=True,
)
