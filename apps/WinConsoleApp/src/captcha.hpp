// apps/cli(Rust)의 C++ 포팅.
// 전처리는 preprocess.rs, 디코딩은 decode.rs를 그대로 옮겼다.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// 모델 옆에 놓이는 사이드카 메타데이터 (<모델>.meta.json).
struct ModelMeta {
	std::string captcha_id;
	int image_width = 0;
	int image_height = 0;
	int label_length = 0;
	std::string characters;  // UTF-8
	int threshold = 255;
	std::string preprocess = "default";

	// 실패 시 std::runtime_error.
	static ModelMeta load(const std::filesystem::path& path);
};

// 이미지 파일 바이트 -> (1, 1, H, W) 순서의 f32 텐서(0.0~1.0).
std::vector<float> preprocess_image(const std::vector<uint8_t>& bytes, const ModelMeta& meta);

// UTF-8 문자열을 코드포인트 단위로 쪼갠다.
std::vector<std::string> split_utf8(const std::string& s);

// (T, N=1, C) 로짓 -> (T, C) log 확률. ONNX 출력은 softmax 이전 값이다.
std::vector<std::vector<double>> log_softmax_frames(const float* logits, size_t num_frames, size_t num_classes);

struct Decoded {
	std::string text;
	double confidence = 0.0;
};

// core.py: ctc_beam_decode_fixed_length. log_probs는 (T, C), 인덱스 0이 blank.
Decoded ctc_beam_decode_fixed_length(const std::vector<std::vector<double>>& log_probs,
                                     const std::vector<std::string>& charset,
                                     size_t expected_length,
                                     size_t beam_width);
