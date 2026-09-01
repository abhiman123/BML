import time
import sys

def print_slow(text, delay=0.03):
    """Prints text with a typewriter effect for dramatic comparison."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def run_simulation():
    print("=" * 65)
    print("       RNN vs. LSTM MEMORY RETENTION SIMULATION")
    print("=" * 65)
    
    print_slow("\n[Task]: Remember the secret code word 'ALPHA' hidden 50 steps ago.")
    print_slow("[Initializing Models...] standard_rnn and lstm_network created.\n")
    time.sleep(1)
    
    # Simulating Standard RNN
    print("--- Running Standard RNN ---")
    steps = [10, 25, 40, 50]
    rnn_memory = "ALPHA"
    
    for step in steps:
        time.sleep(0.6)
        if step < 30:
            rnn_memory = "AL..."
            print(f"Step {step}: Input stream processing... Memory state: '{rnn_memory}' (Fading)")
        else:
            rnn_memory = "???"
            print(f"Step {step}: Input stream processing... Memory state: '{rnn_memory}' (Vanished completely!)")
            
    print_slow("\n-> RNN Result: FAILED to retain long-term context due to vanishing gradients.\n")
    
    time.sleep(1.5)
    
    # Simulating LSTM
    print("--- Running LSTM Network ---")
    lstm_memory = "ALPHA"
    
    for step in steps:
        time.sleep(0.6)
        if step < 40:
            print(f"Step {step}: Input stream processing... Cell State Gate: Active | Memory: 'ALPHA' (Protected)")
        else:
            print(f"Step {step}: Input stream processing... Output Gate: Verified | Memory: 'ALPHA' (Retained!)")
            
    print_slow("\n-> LSTM Result: SUCCESS! Cell state and gates successfully preserved the distant memory.\n")
    
    print("=" * 65)
    print(" SUMMARY: LSTMs outperform RNNs on long sequences because their")
    print(" internal gating mechanism prevents gradients from vanishing.")
    print("=" * 65)

if __name__ == "__main__":
    run_simulation()