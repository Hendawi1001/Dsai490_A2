import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dates based on conditions.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input file (e.g., example_input.txt)")
    parser.add_argument('-o', '--output', required=True, help="Path to save the generated output file")
    
    parser.add_argument('--arch', type=str, default='gru', choices=['gru', 'seq2seq', 'cvae', 'wgan'],
                        help="Which architecture to use for prediction")
    args = parser.parse_args()

    script_path = os.path.join(os.path.dirname(__file__), 'architectures', f'{args.arch}.py')
    
    command = f'python "{script_path}" --predict -i "{args.input}" -o "{args.output}"'
    print(f"Executing: {command}")
    
    os.system(command)