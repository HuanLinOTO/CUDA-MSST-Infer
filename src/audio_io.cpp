#include "audio_io.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif

namespace cudasep {

// ============================================================================
// Helper: read little-endian values from binary stream
// ============================================================================

static uint16_t read_u16(std::ifstream& f) {
    uint8_t buf[2];
    f.read(reinterpret_cast<char*>(buf), 2);
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

static uint32_t read_u32(std::ifstream& f) {
    uint8_t buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}

static void read_tag(std::ifstream& f, char tag[4]) {
    f.read(tag, 4);
}

static bool tag_eq(const char a[4], const char b[4]) {
    return std::memcmp(a, b, 4) == 0;
}

// ============================================================================
// Helper: write little-endian values to binary stream
// ============================================================================

static void write_u16(std::ofstream& f, uint16_t v) {
    uint8_t buf[2] = { (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF) };
    f.write(reinterpret_cast<const char*>(buf), 2);
}

static void write_u32(std::ofstream& f, uint32_t v) {
    uint8_t buf[4] = {
        (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF),
        (uint8_t)((v >> 16) & 0xFF), (uint8_t)((v >> 24) & 0xFF)
    };
    f.write(reinterpret_cast<const char*>(buf), 4);
}

static void write_tag(std::ofstream& f, const char tag[4]) {
    f.write(tag, 4);
}

// ============================================================================
// Helper: get file extension (lowercase)
// ============================================================================

static std::string get_extension(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return ext;
}

// ============================================================================
// WAV Reader
// ============================================================================

AudioData load_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open WAV file: " + path);
    }

    // --- Read RIFF header ---
    char riff_tag[4];
    read_tag(f, riff_tag);
    if (!tag_eq(riff_tag, "RIFF")) {
        throw std::runtime_error("Not a valid WAV file (missing RIFF header): " + path);
    }

    /*uint32_t file_size =*/ read_u32(f); // file size - 8 (not needed)

    char wave_tag[4];
    read_tag(f, wave_tag);
    if (!tag_eq(wave_tag, "WAVE")) {
        throw std::runtime_error("Not a valid WAV file (missing WAVE tag): " + path);
    }

    // --- Scan chunks for "fmt " and "data" ---
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    bool found_fmt = false;
    bool found_data = false;
    uint32_t data_size = 0;
    std::streampos data_offset = 0;

    while (f.good() && !(found_fmt && found_data)) {
        char chunk_id[4];
        read_tag(f, chunk_id);
        if (!f.good()) break;

        uint32_t chunk_size = read_u32(f);
        if (!f.good()) break;

        if (tag_eq(chunk_id, "fmt ")) {
            // Read format chunk
            audio_format = read_u16(f);
            num_channels = read_u16(f);
            sample_rate = read_u32(f);
            /*uint32_t byte_rate =*/ read_u32(f);
            /*uint16_t block_align =*/ read_u16(f);
            bits_per_sample = read_u16(f);

            // Handle extensible format (0xFFFE)
            if (audio_format == 0xFFFE && chunk_size >= 40) {
                // Read cbSize
                uint16_t cb_size = read_u16(f);
                (void)cb_size;
                // valid bits per sample
                /*uint16_t valid_bits =*/ read_u16(f);
                // channel mask
                /*uint32_t channel_mask =*/ read_u32(f);
                // SubFormat GUID: first 2 bytes indicate the actual format
                uint16_t sub_format = read_u16(f);
                // Skip rest of GUID (14 bytes)
                f.seekg(14, std::ios::cur);

                // Map sub_format
                if (sub_format == 1) {
                    audio_format = 1; // PCM
                } else if (sub_format == 3) {
                    audio_format = 3; // IEEE float
                } else {
                    throw std::runtime_error("Unsupported extensible sub-format in WAV: " + std::to_string(sub_format));
                }

                // Skip any remaining bytes in the fmt chunk
                int64_t bytes_read = 16 + 2 + 2 + 4 + 2 + 14; // after base 16 bytes
                int64_t remaining = (int64_t)chunk_size - 16 - bytes_read + 16;
                // Actually, let's compute more carefully:
                // Base fmt fields: 16 bytes (format, channels, sr, byterate, blockalign, bps)
                // Extra: cbSize(2) + validBits(2) + channelMask(4) + subFormat(2) + guid_rest(14) = 24
                // Total read from chunk: 16 + 24 = 40
                int64_t total_read_in_chunk = 40;
                if ((int64_t)chunk_size > total_read_in_chunk) {
                    f.seekg((int64_t)chunk_size - total_read_in_chunk, std::ios::cur);
                }
            } else {
                // Skip extra fmt bytes if chunk_size > 16
                if (chunk_size > 16) {
                    f.seekg((int64_t)(chunk_size - 16), std::ios::cur);
                }
            }
            found_fmt = true;

        } else if (tag_eq(chunk_id, "data")) {
            data_size = chunk_size;
            data_offset = f.tellg();
            found_data = true;
            // Don't read data yet, just mark position

        } else {
            // Unknown chunk — skip it
            // Chunks are padded to even size
            uint32_t skip = chunk_size;
            if (skip & 1) skip++; // pad to even
            f.seekg((int64_t)skip, std::ios::cur);
        }
    }

    if (!found_fmt) {
        throw std::runtime_error("WAV file missing 'fmt ' chunk: " + path);
    }
    if (!found_data) {
        throw std::runtime_error("WAV file missing 'data' chunk: " + path);
    }

    // --- Validate format ---
    if (audio_format != 1 && audio_format != 3) {
        throw std::runtime_error("Unsupported WAV audio format: " + std::to_string(audio_format) +
                                 " (only PCM and IEEE float supported): " + path);
    }

    if (audio_format == 1) {
        if (bits_per_sample != 16 && bits_per_sample != 24 && bits_per_sample != 32) {
            throw std::runtime_error("Unsupported PCM bit depth: " + std::to_string(bits_per_sample) + ": " + path);
        }
    } else if (audio_format == 3) {
        if (bits_per_sample != 32) {
            throw std::runtime_error("Unsupported float bit depth: " + std::to_string(bits_per_sample) + ": " + path);
        }
    }

    int bytes_per_sample = bits_per_sample / 8;
    int64_t total_samples = (int64_t)data_size / bytes_per_sample; // total across all channels
    int64_t num_frames = total_samples / num_channels;

    // --- Read raw sample data ---
    f.seekg(data_offset);
    std::vector<uint8_t> raw(data_size);
    f.read(reinterpret_cast<char*>(raw.data()), data_size);
    if ((size_t)f.gcount() != data_size) {
        throw std::runtime_error("Failed to read all audio data from WAV file: " + path);
    }
    f.close();

    // --- Convert to float and de-interleave (interleaved -> planar) ---
    // Output layout: [channels, num_frames]
    std::vector<float> planar(num_channels * num_frames);

    if (audio_format == 1 && bits_per_sample == 16) {
        // PCM 16-bit signed
        const int16_t* src = reinterpret_cast<const int16_t*>(raw.data());
        for (int64_t i = 0; i < num_frames; i++) {
            for (int ch = 0; ch < num_channels; ch++) {
                planar[ch * num_frames + i] = (float)src[i * num_channels + ch] / 32768.0f;
            }
        }
    } else if (audio_format == 1 && bits_per_sample == 24) {
        // PCM 24-bit signed (3 bytes per sample, little-endian)
        const uint8_t* src = raw.data();
        for (int64_t i = 0; i < num_frames; i++) {
            for (int ch = 0; ch < num_channels; ch++) {
                int64_t idx = (i * num_channels + ch) * 3;
                int32_t sample = (int32_t)src[idx]
                               | ((int32_t)src[idx + 1] << 8)
                               | ((int32_t)src[idx + 2] << 16);
                // Sign-extend from 24 bits
                if (sample & 0x800000) {
                    sample |= (int32_t)0xFF000000;
                }
                planar[ch * num_frames + i] = (float)sample / 8388608.0f;
            }
        }
    } else if (audio_format == 1 && bits_per_sample == 32) {
        // PCM 32-bit signed int
        const int32_t* src = reinterpret_cast<const int32_t*>(raw.data());
        for (int64_t i = 0; i < num_frames; i++) {
            for (int ch = 0; ch < num_channels; ch++) {
                planar[ch * num_frames + i] = (float)src[i * num_channels + ch] / 2147483648.0f;
            }
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        // IEEE float 32-bit
        const float* src = reinterpret_cast<const float*>(raw.data());
        for (int64_t i = 0; i < num_frames; i++) {
            for (int ch = 0; ch < num_channels; ch++) {
                planar[ch * num_frames + i] = src[i * num_channels + ch];
            }
        }
    }

    // --- Upload to GPU ---
    Tensor samples = Tensor::from_cpu_f32(planar.data(), {(int64_t)num_channels, num_frames});

    AudioData result;
    result.samples = std::move(samples);
    result.sample_rate = (int)sample_rate;
    result.channels = (int)num_channels;
    result.num_samples = num_frames;
    return result;
}

// ============================================================================
// WAV Writer
// ============================================================================

void save_wav(const std::string& path, const Tensor& samples, int sample_rate) {
    if (samples.ndim() != 2) {
        throw std::runtime_error("save_wav: expected 2D tensor [channels, num_samples], got " +
                                 std::to_string(samples.ndim()) + "D");
    }
    if (samples.dtype() != DType::Float32) {
        throw std::runtime_error("save_wav: expected Float32 tensor");
    }

    int channels = (int)samples.size(0);
    int64_t num_frames = samples.size(1);

    // --- Copy GPU tensor to CPU (planar layout) ---
    std::vector<float> planar = samples.to_cpu_f32();

    // --- Convert planar [channels, num_frames] to interleaved ---
    std::vector<float> interleaved(channels * num_frames);
    for (int64_t i = 0; i < num_frames; i++) {
        for (int ch = 0; ch < channels; ch++) {
            interleaved[i * channels + ch] = planar[ch * num_frames + i];
        }
    }

    // --- Write WAV file ---
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open output WAV file: " + path);
    }

    uint32_t data_size = (uint32_t)(channels * num_frames * sizeof(float));
    uint32_t fmt_chunk_size = 16;
    uint32_t file_size = 4 + (8 + fmt_chunk_size) + (8 + data_size); // WAVE + fmt chunk + data chunk

    // RIFF header
    write_tag(f, "RIFF");
    write_u32(f, file_size);
    write_tag(f, "WAVE");

    // fmt chunk (IEEE float)
    write_tag(f, "fmt ");
    write_u32(f, fmt_chunk_size);
    write_u16(f, 3);                                               // audio_format = IEEE float
    write_u16(f, (uint16_t)channels);                              // num_channels
    write_u32(f, (uint32_t)sample_rate);                           // sample_rate
    write_u32(f, (uint32_t)(sample_rate * channels * sizeof(float))); // byte_rate
    write_u16(f, (uint16_t)(channels * sizeof(float)));            // block_align
    write_u16(f, 32);                                              // bits_per_sample

    // data chunk
    write_tag(f, "data");
    write_u32(f, data_size);
    f.write(reinterpret_cast<const char*>(interleaved.data()), data_size);

    f.close();
}

// ============================================================================
// FFmpeg fallback
// ============================================================================

AudioData load_via_ffmpeg(const std::string& path) {
    // Generate temp WAV path in same directory as input
    std::string temp_path;
    {
        auto last_sep = path.find_last_of("/\\");
        std::string dir = (last_sep != std::string::npos) ? path.substr(0, last_sep + 1) : "";
        auto last_dot = path.rfind('.');
        std::string base = (last_dot != std::string::npos && last_dot > (last_sep != std::string::npos ? last_sep : 0))
                           ? path.substr((last_sep != std::string::npos ? last_sep + 1 : 0), last_dot - (last_sep != std::string::npos ? last_sep + 1 : 0))
                           : path.substr(last_sep != std::string::npos ? last_sep + 1 : 0);
        temp_path = dir + base + ".tmp.wav";
    }

    // Build ffmpeg command
    // Use pcm_f32le so we get 32-bit float WAV output
#ifdef _WIN32
    std::string cmd = "ffmpeg -y -i \"" + path + "\" -f wav -acodec pcm_f32le \"" + temp_path + "\" 2>nul";
#else
    std::string cmd = "ffmpeg -y -i \"" + path + "\" -f wav -acodec pcm_f32le \"" + temp_path + "\" 2>/dev/null";
#endif

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        // Clean up temp file if it exists
        std::remove(temp_path.c_str());
        throw std::runtime_error("ffmpeg failed to convert audio file: " + path +
                                 " (exit code: " + std::to_string(ret) + "). Is ffmpeg installed and in PATH?");
    }

    // Read the converted WAV
    AudioData result;
    try {
        result = load_wav(temp_path);
    } catch (...) {
        std::remove(temp_path.c_str());
        throw;
    }

    // Clean up temp file
    std::remove(temp_path.c_str());

    return result;
}

// ============================================================================
// Main entry point
// ============================================================================

AudioData load_audio(const std::string& path) {
    std::string ext = get_extension(path);

    if (ext == ".wav" || ext == ".wave") {
        return load_wav(path);
    } else {
        // For all other formats (mp3, flac, ogg, m4a, aac, wma, opus, etc.), use ffmpeg
        return load_via_ffmpeg(path);
    }
}

} // namespace cudasep
