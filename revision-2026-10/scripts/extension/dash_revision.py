"""Single entry point for revision evidence; frozen research engines stay external."""
from pathlib import Path
import argparse,subprocess,sys
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('command',choices=['audit','figures','models','summarize'])
a,extra=p.parse_known_args()
scripts={'audit':'reanalyse.py','figures':'regenerate_figures.py','models':'run_model_extension.py','summarize':'summarize_extension.py'}
raise SystemExit(subprocess.call([sys.executable,str(BASE/scripts[a.command]),*extra]))
