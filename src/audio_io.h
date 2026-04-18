#pragma once
#include "tensor.h"
#include <string>
#include <vector>

namespace cudasep {

struct AudioData {
    Tensor samples;     // [channels, num_samples] Float32 on GPU
    int sample_rate;
    int channels;
    int64_t num_samples;
};

// Read audio file. Returns samples as [channels, num_samples] Float32 Tensor on GPU.
// For WAV: direct parsing. For other formats: uses ffmpeg to convert to WAV first.
AudioData load_audio(const std::string& path);

// Write audio to WAV file. samples: [channels, num_samples] Float32 (on GPU, will be copied to CPU).
void save_wav(const std::string& path, const Tensor& samples, int sample_rate);

// Encode audio to WAV bytes. samples: [channels, num_samples] Float32.
std::vector<uint8_t> encode_wav_bytes(const Tensor& samples, int sample_rate);

// Internal WAV reader
AudioData load_wav(const std::string& path);

// Read via ffmpeg (convert to temp WAV, then read that)
AudioData load_via_ffmpeg(const std::string& path);

} // namespace cudasep
