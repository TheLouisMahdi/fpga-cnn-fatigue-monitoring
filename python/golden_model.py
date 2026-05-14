# golden_model.py

import os
import numpy as np


# ============================================================
# Basic CNN helpers
# ============================================================

def conv2d_3x3(image, kernel, bias=0):
    """
    Generic 3x3 convolution.

    Input:
        image: 2D uint8/int array
        kernel: 3x3 int array

    Output:
        2D int32 array

    For 32x32 input:
        output shape = 30x30
    """
    image = image.astype(np.int32)
    kernel = kernel.astype(np.int32)

    h, w = image.shape
    out = np.zeros((h - 2, w - 2), dtype=np.int32)

    for y in range(h - 2):
        for x in range(w - 2):
            window = image[y:y + 3, x:x + 3]
            out[y, x] = int(np.sum(window * kernel) + bias)

    return out


def relu(x):
    """
    ReLU activation:
        negative values become 0
    """
    return np.maximum(0, x)


def quantize_uint8(x):
    """
    Match FPGA relu_clip.v behavior:

        if x < 0:
            0
        if x > 255:
            255
        else:
            x

    Output dtype:
        uint8
    """
    return np.clip(x, 0, 255).astype(np.uint8)


def maxpool2d_2x2(image):
    """
    2x2 MaxPool with stride 2.

    For 30x30 input:
        output shape = 15x15
    """
    h, w = image.shape
    out = np.zeros((h // 2, w // 2), dtype=np.uint8)

    for y in range(h // 2):
        for x in range(w // 2):
            window = image[y * 2:y * 2 + 2, x * 2:x * 2 + 2]
            out[y, x] = int(np.max(window))

    return out


# ============================================================
# Fixed kernels matching fixed_kernel_bank.v
# ============================================================

def get_fixed_kernels():
    """
    These kernels must match fixed_kernel_bank.v exactly.

    Output order:
        vertical
        horizontal
        diag45
        diag135
        sharpen
        center
    """
    kernels = {
        "vertical": np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ], dtype=np.int32),

        "horizontal": np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
        ], dtype=np.int32),

        "diag45": np.array([
            [0, 1, 1],
            [-1, 0, 1],
            [-1, -1, 0],
        ], dtype=np.int32),

        "diag135": np.array([
            [1, 1, 0],
            [1, 0, -1],
            [0, -1, -1],
        ], dtype=np.int32),

        "sharpen": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ], dtype=np.int32),

        "center": np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.int32),
    }

    return kernels


def summarize_feature_map(pool_uint8):
    """
    Match feature_summary.v behavior.

    For one 15x15 pooled map:
        sum    = sum of all values
        max    = maximum value
        energy = sum(value * value)
    """
    values = pool_uint8.astype(np.uint32).flatten()

    sum_out = int(np.sum(values))
    max_out = int(np.max(values)) if values.size else 0
    energy_out = int(np.sum(values * values))

    return {
        "sum": sum_out,
        "max": max_out,
        "energy": energy_out,
    }


def run_fixed_kernel_feature_extractor(image_32x32):
    """
    Python golden model for the new FPGA architecture.

    Input:
        32x32 grayscale uint8 image

    Pipeline:
        6 fixed kernels
        Conv 3x3
        ReLU + clip to uint8
        MaxPool 2x2
        summary:
            sum
            max
            energy

    Output:
        {
            "features": list of 18 numbers,
            "by_kernel": detailed dictionary
        }

    Feature order matches cnn_feature_extractor_all_top.v:

        0  vertical_sum
        1  vertical_max
        2  vertical_energy

        3  horizontal_sum
        4  horizontal_max
        5  horizontal_energy

        6  diag45_sum
        7  diag45_max
        8  diag45_energy

        9  diag135_sum
        10 diag135_max
        11 diag135_energy

        12 sharpen_sum
        13 sharpen_max
        14 sharpen_energy

        15 center_sum
        16 center_max
        17 center_energy
    """
    image_32x32 = np.asarray(image_32x32)

    if image_32x32.shape != (32, 32):
        raise ValueError(f"Expected 32x32 image, got {image_32x32.shape}")

    image_32x32 = image_32x32.astype(np.uint8)

    kernels = get_fixed_kernels()

    feature_order = [
        "vertical",
        "horizontal",
        "diag45",
        "diag135",
        "sharpen",
        "center",
    ]

    features = []
    by_kernel = {}

    for name in feature_order:
        kernel = kernels[name]

        conv = conv2d_3x3(image_32x32, kernel, bias=0)
        relu_out = relu(conv)
        relu_uint8 = quantize_uint8(relu_out)
        pool = maxpool2d_2x2(relu_uint8)

        summary = summarize_feature_map(pool)

        features.extend([
            summary["sum"],
            summary["max"],
            summary["energy"],
        ])

        by_kernel[name] = {
            "conv": conv,
            "relu_uint8": relu_uint8,
            "pool": pool,
            "sum": summary["sum"],
            "max": summary["max"],
            "energy": summary["energy"],
        }

    return {
        "features": features,
        "by_kernel": by_kernel,
    }


# ============================================================
# File I/O helpers
# ============================================================

def load_txt_image(path, size=32):
    """
    Load decimal TXT image.

    Format:
        one decimal value per line

    Example:
        160
        161
        159
    """
    values = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))

    expected = size * size

    if len(values) != expected:
        raise ValueError(f"{path}: expected {expected} values, got {len(values)}")

    arr = np.array(values, dtype=np.uint8).reshape(size, size)
    return arr


def load_hex_image(path, size=32):
    """
    Load HEX image for FPGA input.

    Format:
        one 8-bit hex value per line

    Example:
        A0
        A1
        9F
    """
    values = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line, 16))

    expected = size * size

    if len(values) != expected:
        raise ValueError(f"{path}: expected {expected} values, got {len(values)}")

    arr = np.array(values, dtype=np.uint8).reshape(size, size)
    return arr


def save_features_txt(features, path):
    """
    Save feature vector as decimal text.

    One feature per line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for value in features:
            f.write(f"{int(value)}\n")


def load_features_txt(path):
    """
    Load feature vector from decimal text.
    """
    values = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))

    return values


def compare_features(fpga_features, python_features):
    """
    Compare FPGA and Python feature vectors.

    Return:
        {
            "match": bool,
            "diffs": list
        }
    """
    if len(fpga_features) != len(python_features):
        return {
            "match": False,
            "diffs": [f"Length mismatch: FPGA={len(fpga_features)}, Python={len(python_features)}"]
        }

    diffs = []

    for i, (f, p) in enumerate(zip(fpga_features, python_features)):
        if int(f) != int(p):
            diffs.append({
                "index": i,
                "fpga": int(f),
                "python": int(p),
                "diff": int(f) - int(p),
            })

    return {
        "match": len(diffs) == 0,
        "diffs": diffs,
    }


# ============================================================
# Legacy compatibility with old main.py
# ============================================================

def get_kernel(mode="identity"):
    """
    Legacy kernel selector.

    Kept so old main.py does not break.
    """
    if mode == "identity":
        return np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.int32)

    if mode == "edge_x":
        return np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.int32)

    if mode == "positive":
        return np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.int32)

    raise ValueError("Unknown kernel mode")


def run_golden_model(image_32x32, kernel_mode="identity"):
    """
    Legacy golden model.

    Old pipeline:
        one selected kernel
        Conv 3x3
        ReLU
        MaxPool
        uint8 output 15x15

    Kept for backward compatibility.
    """
    kernel = get_kernel(kernel_mode)

    conv_out = conv2d_3x3(image_32x32, kernel, bias=0)
    relu_out = relu(conv_out)
    pool_out = maxpool2d_2x2(quantize_uint8(relu_out))

    return {
        "conv": conv_out,
        "relu": relu_out,
        "pool": pool_out
    }


def save_array_txt(array, path):
    """
    Legacy helper.
    """
    flat = np.asarray(array).flatten()

    with open(path, "w", encoding="utf-8") as f:
        for value in flat:
            f.write(f"{int(value)}\n")


def save_array_hex(array, path):
    """
    Legacy helper.
    """
    flat = np.asarray(array).flatten()

    with open(path, "w", encoding="utf-8") as f:
        for value in flat:
            f.write(f"{int(value):02X}\n")


# ============================================================
# Batch generation for current FPGA project
# ============================================================

def generate_python_features_for_rois(mem_dir="mem"):
    """
    Generate Python golden feature files for all three ROIs.

    Input files:
        left_eye.hex
        right_eye.hex
        mouth.hex

    Output files:
        python_features_left_eye.txt
        python_features_right_eye.txt
        python_features_mouth.txt
    """
    roi_files = {
        "left_eye": "left_eye.hex",
        "right_eye": "right_eye.hex",
        "mouth": "mouth.hex",
    }

    outputs = {}

    for roi_name, file_name in roi_files.items():
        input_path = os.path.join(mem_dir, file_name)
        output_path = os.path.join(mem_dir, f"python_features_{roi_name}.txt")

        img = load_hex_image(input_path, size=32)
        result = run_fixed_kernel_feature_extractor(img)

        save_features_txt(result["features"], output_path)

        outputs[roi_name] = {
            "input": input_path,
            "output": output_path,
            "features": result["features"],
        }

    return outputs


def compare_fpga_with_python(mem_dir="mem"):
    """
    Compare ModelSim FPGA output files with Python golden files.

    FPGA files:
        fpga_features_left_eye.txt
        fpga_features_right_eye.txt
        fpga_features_mouth.txt

    Python files:
        python_features_left_eye.txt
        python_features_right_eye.txt
        python_features_mouth.txt
    """
    roi_names = ["left_eye", "right_eye", "mouth"]

    report = {}

    for roi_name in roi_names:
        fpga_path = os.path.join(mem_dir, f"fpga_features_{roi_name}.txt")
        python_path = os.path.join(mem_dir, f"python_features_{roi_name}.txt")

        fpga_features = load_features_txt(fpga_path)
        python_features = load_features_txt(python_path)

        report[roi_name] = compare_features(fpga_features, python_features)

    return report


# ============================================================
# Main test
# ============================================================

if __name__ == "__main__":
    MEM_DIR = r"C:\Users\poimu\Desktop\cnn_fpga_project\FPGAp\mem"

    print("Generating Python golden features...")
    outputs = generate_python_features_for_rois(MEM_DIR)

    for roi_name, info in outputs.items():
        print(f"{roi_name}:")
        print(f"  input : {info['input']}")
        print(f"  output: {info['output']}")
        print(f"  features:")
        for i, value in enumerate(info["features"]):
            print(f"    feature[{i:02d}] = {value}")

    print("\nComparing FPGA outputs with Python golden outputs...")

    try:
        report = compare_fpga_with_python(MEM_DIR)

        for roi_name, result in report.items():
            if result["match"]:
                print(f"{roi_name}: MATCH")
            else:
                print(f"{roi_name}: MISMATCH")
                for diff in result["diffs"][:20]:
                    print(
                        f"  index={diff['index']} "
                        f"fpga={diff['fpga']} "
                        f"python={diff['python']} "
                        f"diff={diff['diff']}"
                    )

                if len(result["diffs"]) > 20:
                    print(f"  ... {len(result['diffs']) - 20} more diffs")

    except FileNotFoundError:
        print("FPGA output files not found yet.")
        print("Run ModelSim first to create fpga_features_*.txt files.")