import argparse
from PSZS.Models.funcs import mixing_progress

def main(args):
    max_steps = getattr(args, 'max-steps', 100)
    strategies = getattr(args, 'strategies', ['curriculum', 'step', 'exponential', 'fixed', 'linear'])
    if strategies is None:
        strategies = ['curriculum', 'step', 'exponential', 'fixed', 'linear']
    for i in range(max_steps):
        print(f"Step {i}: "+ " | ".join([f"{strategy}: {str(mixing_progress(strategy, i, max_steps))}" for strategy in strategies]))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=100, help='Number of steps')
    parser.add_argument('--strategies', type=str, nargs='*', help='Mixing strategies to test')
    args = parser.parse_args()
    main(args)