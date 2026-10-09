import os,sys
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1";sys.dont_write_bytecode=True
def main():
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    for command,keys in {"run":("method","source-root","data-root","output-root","derived-root","gate"),
                         "synthetic":("method","source-root","derived-root","output-root"),
                         "replay":("folder","source-root","data-root","derived-root")}.items():
        parser=sub.add_parser(command)
        for key in keys:parser.add_argument("--"+key,required=True)
    a=p.parse_args()
    if a.command=="run":
        from .runner import run_once
        run_once(a.method,a.source_root,a.data_root,a.output_root,a.derived_root,a.gate)
    elif a.command=="synthetic":
        from .synthetic import validate
        validate(a.method,a.source_root,a.derived_root,a.output_root)
    else:
        from .runner import replay_artifact
        replay_artifact(a.folder,a.source_root,a.data_root,a.derived_root)
if __name__=="__main__":main()
