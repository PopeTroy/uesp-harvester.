import os
import shutil
import time
import numpy as np

# ==========================================
# 1. LOCAL FILESYSTEM CONTROLLER
# ==========================================
class LocalFileSystem:
    """Provides local file operations without cloud reliance."""

    @staticmethod
    def mkdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 Created directory: {folder_path}")

    @staticmethod
    def write(file_path, content):
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"📝 Written to: {file_path}")

    @staticmethod
    def read(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def delete(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"🗑️ Deleted directory: {path}")
        elif os.path.exists(path):
            os.remove(path)
            print(f"🗑️ Deleted file: {path}")

    @staticmethod
    def stat(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        info = os.stat(path)
        return {
            "size_bytes": info.st_size,
            "created_time": time.ctime(info.st_ctime),
            "modified_time": time.ctime(info.st_mtime),
            "is_directory": os.path.isdir(path)
        }

    @staticmethod
    def copy(src_path, dest_path):
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest_path)
        print(f"📋 Copied: {src_path} ➔ {dest_path}")

    @staticmethod
    def paste(src_path, dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)
        LocalFileSystem.copy(src_path, dest_path)


# ==========================================
# 2. DIGITAL IMAGE PROCESSING (DIP) CORE
# ==========================================
class LocalDIPCore:
    """Executes array-based Digital Image Processing locally."""

    @staticmethod
    def apply_contrast_stretching(image_matrix):
        """Normalize pixel intensity range [0, 255]."""
        min_val, max_val = np.min(image_matrix), np.max(image_matrix)
        if max_val == min_val:
            return image_matrix
        stretched = ((image_matrix - min_val) / (max_val - min_val)) * 255.0
        return stretched.astype(np.uint8)

    @staticmethod
    def compute_edge_density(image_matrix):
        """Local spatial gradient extraction."""
        dx = np.abs(np.diff(image_matrix, axis=1))
        return float(np.mean(dx))


# ==========================================
# 3. PID CONTROLLER CORE
# ==========================================
class PIDController:
    """Discrete Proportional-Integral-Derivative controller."""

    def __init__(self, Kp=1.5, Ki=0.2, Kd=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measured_value, dt=0.1):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output


# ==========================================
# 4. ML ONLINE ADAPTATION CORE
# ==========================================
class LocalMLAdapter:
    """Simple stochastic gradient update model to auto-tune PID gains."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def tune_pid(self, pid_instance, performance_error):
        # Dynamically adjust proportional gain based on systemic error rate
        pid_instance.Kp += self.lr * performance_error


# ==========================================
# 5. INTEGRATED EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Define local documents directory path
    DOCS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "UESP_Local_Engine")
    
    fs = LocalFileSystem()
    dip = LocalDIPCore()
    pid = PIDController(Kp=1.2, Ki=0.1, Kd=0.02)
    ml = LocalMLAdapter(lr=0.005)

    # 1. Directory Structure Operations
    fs.mkdir(DOCS_DIR)
    file_a = os.path.join(DOCS_DIR, "raw_sensor_log.txt")
    fs.write(file_a, "SESSION_STATE: ACTIVE\nSAMPLE_RATE: 100Hz")
    
    print("\n--- File Stat Info ---")
    print(fs.stat(file_a))

    # 2. Copy/Paste Operations
    backup_dir = os.path.join(DOCS_DIR, "Backups")
    fs.paste(file_a, backup_dir)

    # 3. DIP Matrix Processing
    raw_synthetic_image = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
    processed_image = dip.apply_contrast_stretching(raw_synthetic_image)
    edge_score = dip.compute_edge_density(processed_image)
    print(f"\n📷 Processed Image Edge Density: {edge_score:.4f}")

    # 4. ML-Guided PID Loop Simulation
    setpoint = 100.0
    current_value = 20.0
    
    print("\n--- Running Local PID & ML Optimization Loop ---")
    for step in range(5):
        control_output = pid.compute(setpoint, current_value)
        current_value += control_output * 0.2  # Simulate system response
        error = setpoint - current_value
        
        # ML model adjusts PID gain based on tracking error
        ml.tune_pid(pid, error)
        
        print(f"Step {step+1}: Measured={current_value:.2f} | Control Out={control_output:.2f} | Updated Kp={pid.Kp:.4f}")

    # Log results locally
    results_path = os.path.join(DOCS_DIR, "execution_summary.txt")
    fs.write(results_path, f"Final Value: {current_value}\nFinal Kp: {pid.Kp}\nEdge Density: {edge_score}")
    
    print(f"\n✅ All operations completed locally. Check directory: {DOCS_DIR}")
