import argparse
import json
import os
import subprocess
from pathlib import Path

from adversarial_testbed import ExperimentOrchestrator
from adversarial_testbed.factories import create_dataset


def run_chunks(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.chunk_id is not None:
        config["chunk_id"] = args.chunk_id

    if args.chunk_size is not None:
        config["chunk_size"] = args.chunk_size

    if "SLURM_ARRAY_TASK_ID" in os.environ and args.chunk_id is None:
        config["chunk_id"] = int(os.environ["SLURM_ARRAY_TASK_ID"])

    orchestrator = ExperimentOrchestrator(config)

    output_path = orchestrator.run()
    print(f"Chunk execution complete. Results saved to: {output_path}")


def submit_slurm(args):
    """Inspects dataset, calculates chunks, and generates/submits a Slurm job array."""
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset = create_dataset(config["dataset"])
    total_items = len(dataset)
    chunk_size = args.chunk_size

    num_chunks = (total_items + chunk_size - 1) // chunk_size
    max_array_idx = num_chunks - 1

    print(f"Dataset total items: {total_items}")
    print(
        f"Chunk size: {chunk_size} -> Generating {num_chunks} total tasks (Array: 0-{max_array_idx})"
    )

    slurm_script_content = f"""
        #!/bin/bash
        #SBATCH --job-name={config.get("experiment_id", "adversarial_exp")}
        #SBATCH --output=logs/slurm_%A_%a.out
        #SBATCH --error=logs/slurm_%A_%a.err
        #SBATCH --array=0-{max_array_idx}
        #SBATCH --time={args.time}
        #SBATCH --cpus-per-task={args.cpus}
        #SBATCH --gres=gpu:{args.gpus}
        #SBATCH --mem={args.mem}

        echo "Running task $SLURM_ARRAY_TASK_ID on$(hostname)"

        uv run {os.path.abspath(__file__)} run \\
            --config {os.path.abspath(args.config)} \\
            --chunk-id $SLURM_ARRAY_TASK_ID \\
            --chunk-size {chunk_size}
    """

    Path("logs").mkdir(exist_ok=True)
    script_path = Path("submit_job.sh")
    script_path.write_text(slurm_script_content)
    print(f"Generated Slurm batch script at: {script_path}")

    if args.submit:
        result = subprocess.run(  # noqa: PLW1510
            ["sbatch", str(script_path)], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Successfully submitted job array! Output:\n{result.stdout.strip()}")
        else:
            print(f"Error submitting job:\n{result.stderr.strip()}")
    else:
        print(
            "Dry run mode: Script generated but not submitted. Run `sbatch submit_job.sh` manually when ready."
        )


def main():
    parser = argparse.ArgumentParser(description="HPC Experiment Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: Run a single chunk
    run_parser = subparsers.add_parser("run", help="Run a single dataset chunk.")
    run_parser.add_argument("--config", required=True, help="Path to JSON config file.")
    run_parser.add_argument(
        "--chunk-id", type=int, help="Index of the chunk to process."
    )
    run_parser.add_argument("--chunk-size", type=int, help="Number of items per chunk.")
    run_parser.set_defaults(func=run_chunks)

    # Subcommand: Submit via Slurm
    submit_parser = subparsers.add_parser(
        "submit", help="Generate and submit Slurm job arrays."
    )
    submit_parser.add_argument(
        "--config", required=True, help="Path to JSON config file."
    )
    submit_parser.add_argument(
        "--chunk-size", type=int, default=50, help="Items per chunk/job."
    )
    submit_parser.add_argument("--time", default="02:00:00", help="Slurm time limit.")
    submit_parser.add_argument("--cpus", type=int, default=4, help="CPUs per task.")
    submit_parser.add_argument("--gpus", type=int, default=1, help="GPUs per task.")
    submit_parser.add_argument("--mem", default="16G", help="Memory per task.")
    submit_parser.add_argument(
        "--submit",
        action="store_true",
        help="Instantly call sbatch after generating script.",
    )
    submit_parser.set_defaults(func=submit_slurm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
