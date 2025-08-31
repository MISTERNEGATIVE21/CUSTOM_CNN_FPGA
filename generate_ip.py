import hls4ml
import onnx
import os
import pprint

# --- Pre-defined Configurations ---

# Board-specific details are now selectable in the menu
TARGET_BOARDS = {
    "1": {
        "name": "Digilent Zybo Z7-10",
        "part": "xc7z010-clg400-1"
    },
    "2": {
        "name": "Avnet ZedBoard", # ZedBoard with Zynq 7020
        "part": "xc7z020-clg484-1" # Correct part number for ZedBoard
    }
    # You can easily add more boards here in the future
}

def get_resource_optimized_config(onnx_model, board_part):
    """
    Configuration that prioritizes using fewer FPGA resources.
    Good for large models on smaller FPGAs.
    """
    config = hls4ml.utils.config_from_onnx_model(onnx_model, granularity='name')
    config['OutputDir'] = f'hls4ml_ip_{board_part}_resource_opt'
    config['ProjectName'] = 'my_model_ip_resource'
    config['IOType'] = 'io_stream' # AXI4-Stream is best for Zynq-based systems
    config['HLSConfig'] = {
        'Model': {
            'Precision': 'ap_fixed<16,6>',
            'ReuseFactor': 8, # High reuse factor saves a lot of DSPs/LUTs
            'Strategy': 'Resource' # Tell HLS to prioritize low resource usage
        }
    }
    config['FPGA'] = {'Part': board_part}
    config['ClockPeriod'] = 10 # 100 MHz is a reasonable target
    return config

def get_performance_optimized_config(onnx_model, board_part):
    """
    Configuration that prioritizes low latency (high performance).
    Uses more FPGA resources.
    """
    config = hls4ml.utils.config_from_onnx_model(onnx_model, granularity='name')
    config['OutputDir'] = f'hls4ml_ip_{board_part}_performance_opt'
    config['ProjectName'] = 'my_model_ip_performance'
    config['IOType'] = 'io_stream'
    config['HLSConfig'] = {
        'Model': {
            'Precision': 'ap_fixed<16,6>',
            'ReuseFactor': 1, # No reuse = full parallelism = lowest latency
            'Strategy': 'Latency' # Tell HLS to prioritize low latency
        }
    }
    config['FPGA'] = {'Part': board_part}
    config['ClockPeriod'] = 10 # 100 MHz
    return config

# --- Menu Functions ---

def select_from_menu(options_dict, title):
    """Generic function to display a menu and get user's choice."""
    print(f"\n--- {title} ---")
    for key, value in options_dict.items():
        if isinstance(value, dict):
            print(f"  [{key}] {value['name']}")
        else:
            print(f"  [{key}] {value}")

    while True:
        choice = input("Your choice: ")
        if choice in options_dict:
            return options_dict[choice]
        else:
            print("Invalid choice. Please try again.")


def main():
    """Main function to drive the menu and conversion process."""
    print("======================================================")
    print("      hls4ml ONNX to IP Core Generation Script      ")
    print("======================================================")

    # 1. Select ONNX Model
    onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
    if not onnx_files:
        print("\n❌ Error: No .onnx files found in the current directory. Exiting.")
        return

    onnx_file_map = {str(i+1): f for i, f in enumerate(onnx_files)}
    selected_model_file = select_from_menu(onnx_file_map, "STEP 1: Select an ONNX Model")
    print(f"✅ Model selected: {selected_model_file}")
    onnx_model = onnx.load(selected_model_file)

    # 2. Select Target Board
    selected_board = select_from_menu(TARGET_BOARDS, "STEP 2: Select a Target Board")
    print(f"✅ Board selected: {selected_board['name']}")

    # 3. Select Configuration Strategy
    strategies = {
        "1": "Resource Optimized (Low FPGA Usage)",
        "2": "Performance Optimized (Low Latency)"
    }
    selected_strategy = select_from_menu(strategies, "STEP 3: Select a Configuration Strategy")

    if selected_strategy == strategies["1"]:
        config = get_resource_optimized_config(onnx_model, selected_board['part'])
        print("✅ Strategy: Resource Optimized")
    else:
        config = get_performance_optimized_config(onnx_model, selected_board['part'])
        print("✅ Strategy: Performance Optimized")

    # 4. Review Configuration and Confirm
    print("\n--- STEP 4: Review Final Configuration ---")
    pprint.pprint(config)

    while True:
        confirm = input("\nProceed with this configuration? (yes/no): ").lower()
        if confirm in ["yes", "y"]:
            break
        elif confirm in ["no", "n"]:
            print("Operation cancelled. Exiting.")
            return
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    # 5. Run Conversion and Build
    print("\n--- STEP 5: Running HLS Conversion and Build ---")
    print("This process may take several minutes...")

    try:
        hls_model = hls4ml.converters.convert_from_onnx_model(
            onnx_model,
            hls_config=config
        )
        hls_model.compile()
        hls_model.build(csim=False, export=True)

        print("\n======================================================")
        print("🎉 SUCCESS! IP Core generated.")
        print(f"Find your HLS project and IP core in the '{config['OutputDir']}' directory.")
        print("======================================================")

    except Exception as e:
        print(f"\n❌ An error occurred during the build process: {e}")


if __name__ == "__main__":
    main()
