import argparse
from tah.evaluate.eval_unified import allocate_gpus_and_run_jobs

def main(args):
    """Main coordinator function"""
    allocate_gpus_and_run_jobs(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM inference and evaluation with multiple backends")
    parser.add_argument("--eval_config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--backend", type=str, choices=['sglang', 'hf', 'tah'], default='hf',
                       help="Inference backend to use: 'sglang', 'hf', or 'tah' (default: hf)")
    
    # Add job-based processing arguments
    parser.add_argument('--job_nums', type=int, default=1,
                      help='Total number of jobs to split the dataset into')
    parser.add_argument('--tp_size_per_job', type=int, default=1,
                      help='Number of GPUs (tensor parallel size) per job')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to the model')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Path to the output directory, default is model_path')
    parser.add_argument('--dataset_name', type=str, default=None,
                      help='Name of the dataset to use (supports multiple datasets separated by commas, e.g., "aime24,math500")')
    parser.add_argument('--data_range', type=int, nargs='+', default=None,
                      help='Data range: either [end] or [start, end]')
    parser.add_argument('--data_ids', type=str, default=None,
                      help='Comma-separated indices to evaluate, e.g., "0,5,6,15". If provided, overrides --data_range')
    parser.add_argument('--del_job_dir', type=bool, default=True,
                      help='Delete job directory after evaluation')
    
    args = parser.parse_args()
    
    main(args)
