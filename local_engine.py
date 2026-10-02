import os
import shutil
import time
import json
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
# 3. TEACHER-STUDENT DISTILLATION MODULE
# ==========================================
class TeacherStudentDistillation:
    """Distill heavy teacher target outputs into a lightweight local student model."""

    def __init__(self, storage_path, temperature=2.0):
        self.storage_path = storage_path
        self.temperature = temperature
        self.fs = LocalFileSystem()
        # Initialize small local weights (student linear model)
        self.student_weights = np.array([0.5, 0.1], dtype=np.float32)

    def teacher_policy(self, current_speed, target_speed):
        """Simulates a complex, heavy teacher network recommending target velocity adjustments."""
        error = target_speed - current_speed
        # Non-linear teacher evaluation (e.g. high-capacity model prediction)
        suggested_adjustment = np.tanh(error / self.temperature) * 15.0
        return float(suggested_adjustment)

    def predict_student(self, current_speed, target_speed):
        """Lightweight student model forward pass."""
        inputs = np.array([target_speed - current_speed, current_speed / 100.0], dtype=np.float32)
        return float(np.dot(self.student_weights, inputs))

    def train_step(self, current_speed, target_speed, lr=0.01):
        """Knowledge distillation: Student updates its parameters toward Teacher output."""
        teacher_out = self.teacher_policy(current_speed, target_speed)
        student_out = self.predict_student(current_speed, target_speed)
        
        # Mean Squared Error distillation loss gradient
        loss = (student_out - teacher_out) ** 2
        grad = 2.0 * (student_out - teacher_out)
        
        inputs = np.array([target_speed - current_speed, current_speed / 100.0], dtype=np.float32)
        self.student_weights -= lr * grad * inputs
        
        return teacher_out, student_out, loss

    def save_distillation_state(self, step, teacher_out, student_out, loss):
        """Persists the distilled state and weights to disk."""
        data = {
            "step": step,
            "timestamp": time.time(),
            "student_weights": self.student_weights.tolist(),
            "teacher_output": teacher_out,
            "student_output": student_out,
            "distillation_loss": loss
        }
        self.fs.write(self.storage_path, json.dumps(data, indent=2))


# ==========================================
# 4. PID CONTROLLER CORE (CRUISE CONTROL)
# ==========================================
class PIDController:
    """Discrete Proportional-Integral-Derivative controller for smooth trajectory execution."""

    def __init__(self, Kp=0.8, Ki=0.15, Kd=0.05, max_accel=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_accel = max_accel
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measured_value, dt=0.1):
        error = setpoint - measured_value
        self.integral += error * dt
        # Prevent integral windup
        self.integral = np.clip(self.integral, -20.0, 20.0)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # Smooth cruise control bounds (limits abrupt spikes)
        output = np.clip(output, -self.max_accel, self.max_accel)
        self.prev_error = error
        return float(output)


# ==========================================
# 5. ML ONLINE ADAPTATION CORE
# ==========================================
class LocalMLAdapter:
    """Simple stochastic gradient update model to auto-tune PID gains."""

    def __init__(self, lr=0.001):
        self.lr = lr

    def tune_pid(self, pid_instance, tracking_error):
        # Dynamically adjust proportional gain based on systemic error rate
        pid_instance.Kp += self.lr * abs(tracking_error)
        pid_instance.Kp = np.clip(pid_instance.Kp, 0.1, 3.0)


# ==========================================
# 6. INTEGRATED EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    DOCS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "UESP_Local_Engine")
    
    fs = LocalFileSystem()
    dip = LocalDIPCore()
    pid = PIDController(Kp=0.8, Ki=0.15, Kd=0.05, max_accel=12.0)
    ml = LocalMLAdapter(lr=0.002)
    
    distill_db_path = os.path.join(DOCS_DIR, "distillation_memory.json")
    distiller = TeacherStudentDistillation(storage_path=distill_db_path)

    # 1. Setup Local Storage
    fs.mkdir(DOCS_DIR)
    log_file = os.path.join(DOCS_DIR, "raw_sensor_log.txt")
    fs.write(log_file, "CRUISE_CONTROL_STATE: ACTIVE\nENGINE: UESP_DISTILLED_PID\n")

    # 2. DIP Matrix Processing
    raw_synthetic_image = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
    processed_image = dip.apply_contrast_stretching(raw_synthetic_image)
    edge_score = dip.compute_edge_density(processed_image)
    print(f"\n📷 DIP Processed Edge Density: {edge_score:.4f}")

    # 3. Teacher-Student Distillation & Smooth PID Cruise Loop
    target_speed = 120.0  # Cruise Setpoint (km/h)
    current_speed = 40.0  # Starting Speed (km/h)
    
    print("\n--- Running Teacher-Student Distillation & PID Cruise Control ---")
    history_logs = []

    for step in range(1, 11):
        # A. Distill teacher knowledge to lightweight student model
        teacher_adj, student_adj, dist_loss = distiller.train_step(current_speed, target_speed)
        
        # B. Combine cruise setpoint with distilled student output recommendations
        effective_target = target_speed + student_adj
        
        # C. Pass effective target into PID Controller for smooth acceleration/braking
        accel_command = pid.compute(effective_target, current_speed, dt=0.2)
        
        # Simulate physical vehicle response
        current_speed += accel_command * 0.5
        tracking_error = target_speed - current_speed
        
        # D. ML Auto-tuner adjusts PID parameters online
        ml.tune_pid(pid, tracking_error)
        
        # E. Persist Distillation & Telemetry locally
        distiller.save_distillation_state(step, teacher_adj, student_adj, dist_loss)
        
        log_entry = (
            f"Step {step:02d} | Current Speed: {current_speed:6.2f} km/h | "
            f"Teacher Out: {teacher_adj:6.2f} | Student Out: {student_adj:6.2f} | "
            f"PID Accel Cmd: {accel_command:6.2f} | Loss: {dist_loss:.4f}"
        )
        print(log_entry)
        history_logs.append(log_entry)

    # Log summary locally
    summary_path = os.path.join(DOCS_DIR, "execution_summary.txt")
    summary_content = (
        f"--- CRUISE CONTROL EXECUTION SUMMARY ---\n"
        f"Final Speed: {current_speed:.2f} km/h\n"
        f"Final PID Kp: {pid.Kp:.4f}\n"
        f"Distillation Storage Path: {distill_db_path}\n"
        f"DIP Edge Density: {edge_score:.4f}\n\n"
        + "\n".join(history_logs)
    )
    fs.write(summary_path, summary_content)
    
    print(f"\n✅ All operations completed locally. Telemetry and state saved in: {DOCS_DIR}")
