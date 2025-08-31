import hls4ml
import tensorflow as tf
import os
import pprint

# --- Pre-defined Board and Configuration Settings ---

# Board-specific part numbers
TARGET_BOARDS = {
    "1": {
        "name": "Digilent Zybo Z7-10",
        "part": "xc7z010-clg400-1"
    },
    "2": {
        "name": "Avnet ZedBoard",
        "part": "xc7z020-clg404-1"
    }
}

def get_resource_optimized_config(keras_model, board_part):
    """
    Generates an hls4ml config prioritizing low FPGA resource usage.
    """
    config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    config['Model']['ReuseFactor'] = 16  # High reuse factor saves a lot of resources
    config['Model']['Strategy'] = 'Resource'
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    config['IOType'] = 'io_stream'  # AXI4-Stream is best for Zynq systems
    config['HLSConfig']['FPGA'] = {'Part': board_part}
    config['HLSConfig']['ClockPeriod'] = 10  # 100 MHz clock
    config['OutputDir'] = f'hls4ml_ip_{board_part}_resource_opt'
    config['ProjectName'] = 'my_keras_model_ip_resource'
    return config

def get_performance_optimized_config(keras_model, board_part):
    """
    Generates an hls4ml config prioritizing low latency (high performance).
    """
    config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    config['Model']['ReuseFactor'] = 1  # No reuse = full parallelism = lowest latency
    config['Model']['Strategy'] = 'Latency'
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    config['IOType'] = 'io_stream'
    config['HLSConfig']['FPGA'] = {'Part': board_part}
    config['HLSConfig']['ClockPeriod'] = 10
    config['OutputDir'] = f'hls4ml_ip_{board_part}_performance_opt'
    config['ProjectName'] = 'my_keras_model_ip_performance'
    return config

# --- Menu Helper ---

def select_from_menu(options_dict, title):
    """Generic function to display a menu and get the user's choice."""
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

# --- Main Workflow ---

def main():
    """Main function to drive the menu and conversion process."""
    print("======================================================")
    print("   hls4ml Keras (.h5) to IP Core Generation Script    ")
    print("======================================================")

    # 1. Select Keras Model
    h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    if not h5_files:
        print("\n❌ Error: No .h5 model files found in the current directory. Exiting.")
        return

    h5_file_map = {str(i + 1): f for i, f in enumerate(h5_files)}
    selected_model_file = select_from_menu(h5_file_map, "STEP 1: Select a Keras Model (.h5)")
    print(f"✅ Model selected: {selected_model_file}")

    # Load the Keras model with custom objects for QKeras layers
    print("Loading Keras model...")
    # --- FIX APPLIED HERE ---
    # The get_qkeras_custom_objects function was moved to the keras_utils submodule
    keras_model = tf.keras.models.load_model(
        selected_model_file,
        custom_objects=hls4ml.utils.keras_utils.get_qkeras_custom_objects()
    )
    print("Model loaded successfully.")

    # 2. Select Target Board
    selected_board = select_from_menu(TARGET_BOARDS, "STEP 2: Select a Target Board")
    print(f"✅ Board selected: {selected_board['name']}")

    # 3. Select Configuration Strategy
    strategies = {
        "1": "Resource Optimized (Low FPGA Usage)",
        "2": "Performance Optimized (Low Latency)"
    }
    selected_strategy_name = select_from_menu(strategies, "STEP 3: Select an Optimization Strategy")

    if selected_strategy_name == strategies["1"]:
        config = get_resource_optimized_config(keras_model, selected_board['part'])
        print("✅ Strategy: Resource Optimized")
    else:
        config = get_performance_optimized_config(keras_model, selected_board['part'])
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
    print("\n--- STEP 5: Converting Keras model to HLS project ---")
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=config,
        output_dir=config['OutputDir'],
        project_name=config['ProjectName']
    )

    print("\n--- Compiling HLS project ---")
    hls_model.compile()

    print("\n--- Building HLS project to generate IP core... (This may take several minutes) ---")
    hls_model.build(export=True)

    print("\n======================================================")
    print("🎉 SUCCESS! IP Core generated.")
    print(f"Find your HLS project and IP core in the '{config['OutputDir']}' directory.")
    print("======================================================")


if __name__ == "__main__":
    main()
