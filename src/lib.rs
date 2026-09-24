use pyo3::prelude::*;
use sha2::{Sha256, Digest};
use chrono::Utc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[pyfunction]
fn generate_ecta_session_id(user_payload: &str) -> PyResult<String> {
    let now = Utc::now();
    let ecta_timestamp = now.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string();
    let dilated_quantum_ticks = now.timestamp_nanos_opt().unwrap_or(0) * 6000;
    let raw_seed = format!("ECTA_SEC58_VALIDATED::{}:{}:DILATION_6000::{}", ecta_timestamp, dilated_quantum_ticks, user_payload);
    let mut hasher = Sha256::new();
    hasher.update(raw_seed.as_bytes());
    let hash_result = hasher.finalize();
    Ok(format!("UESP-SESSION-{:x}", hash_result))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_transform_impl(data: &[f32], factor: f32) -> Vec<f32> {
    let len = data.len();
    let mut output = vec![0.0f32; len];
    
    // SE(3) Spatial Invariance Step: Center of Mass Normalization
    let sum: f32 = data.iter().sum();
    let mean = if len > 0 { sum / len as f32 } else { 0.0 };

    let mut i = 0;
    let factor_vec = _mm256_set1_ps(factor);
    let mean_vec = _mm256_set1_ps(mean);

    while i + 8 <= len {
        // Load 8 float32 elements
        let vec_in = _mm256_loadu_ps(data.as_ptr().add(i));
        
        // Zero-center the tensor vector (Translation Invariance)
        let vec_centered = _mm256_sub_ps(vec_in, mean_vec);
        
        // Multiply by Quantum Dilation Factor (6000.0)
        let vec_out = _mm256_mul_ps(vec_centered, factor_vec);
        
        _mm256_storeu_ps(output.as_mut_ptr().add(i), vec_out);
        i += 8;
    }

    while i < len {
        output[i] = (data[i] - mean) * factor;
        i += 1;
    }

    output
}

#[pyfunction]
fn avx2_quantum_tensor_transform(data: Vec<f32>) -> PyResult<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { avx2_transform_impl(&data, 6000.0) });
        }
    }

    // Baseline Fallback with Zero-Centering (SE(3) Spatial Invariance)
    let len = data.len();
    let sum: f32 = data.iter().sum();
    let mean = if len > 0 { sum / len as f32 } else { 0.0 };

    Ok(data.iter().map(|x| (x - mean) * 6000.0).collect())
}

#[pymodule]
fn uesp_quantum_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_ecta_session_id, m)?)?;
    m.add_function(wrap_pyfunction!(avx2_quantum_tensor_transform, m)?)?;
    Ok(())
}
