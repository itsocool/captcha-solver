// ConsoleApp.v4/Program.cs의 포팅. 인자 문법과 출력 형식을 그대로 맞췄다.
//
//   captcha -c="gov24" -i="gov24.JPG" [-m="gov24.model"]
//
// 예측 문자열만 개행 없이 stdout으로 나가고, 오류는 stderr + 0이 아닌 종료 코드다.
//
// 경로는 std::filesystem::path 로만 다룬다. path::value_type 이 Windows에서 wchar_t,
// POSIX에서 char 라 ONNX Runtime의 ORTCHAR_T 와 정확히 일치하므로 변환이 필요 없다.

#include "captcha.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <climits>
#include <unistd.h>
#endif

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

// Windows: wchar_t, POSIX: char. ORT_TSTR()은 ONNX Runtime 헤더가 준다.
using Str = fs::path::string_type;
using Chr = fs::path::value_type;

namespace {

// ---------------------------------------------------------------- 플랫폼 의존

// 오류 메시지에 경로를 찍기 위한 변환.
std::string narrow(const fs::path& p) {
#ifdef _WIN32
	// 콘솔 코드페이지로 맞춰야 한글 경로가 깨지지 않는다.
	const std::wstring& s = p.native();
	UINT cp = GetConsoleOutputCP();
	if (cp == 0) cp = CP_ACP;
	const int n = WideCharToMultiByte(cp, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
	std::string out(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(cp, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n, nullptr, nullptr);
	return out;
#else
	return p.native();
#endif
}

fs::path exe_dir() {
#ifdef _WIN32
	std::wstring buf(MAX_PATH, L'\0');
	for (;;) {
		const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
		if (n == 0) return fs::current_path();
		if (n < buf.size()) {
			buf.resize(n);
			break;
		}
		buf.resize(buf.size() * 2);
	}
	return fs::path(buf).parent_path();
#else
	char buf[PATH_MAX];
	const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	if (n <= 0) return fs::current_path();
	buf[n] = '\0';
	return fs::path(buf).parent_path();
#endif
}

// ---------------------------------------------------------------- 유틸

bool file_exists(const fs::path& path) {
	std::error_code ec;
	return fs::is_regular_file(path, ec);
}

std::vector<uint8_t> read_file(const fs::path& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) throw std::runtime_error("cannot open file: " + narrow(path));
	return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------- 최소 JSON
// meta.json은 평평한 객체(문자열/숫자만)라 이것만 다룬다. 중첩이 나오면 에러.

void append_utf8(std::string& out, unsigned cp) {
	if (cp < 0x80) {
		out += static_cast<char>(cp);
	} else if (cp < 0x800) {
		out += static_cast<char>(0xC0 | (cp >> 6));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	} else if (cp < 0x10000) {
		out += static_cast<char>(0xE0 | (cp >> 12));
		out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	} else {
		out += static_cast<char>(0xF0 | (cp >> 18));
		out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
		out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		out += static_cast<char>(0x80 | (cp & 0x3F));
	}
}

void skip_ws(const std::string& s, size_t& i) {
	while (i < s.size() && static_cast<unsigned char>(s[i]) <= ' ') ++i;
}

std::string parse_json_string(const std::string& s, size_t& i) {
	if (i >= s.size() || s[i] != '"') throw std::runtime_error("meta.json: 문자열이 와야 합니다");
	++i;
	std::string out;
	while (i < s.size() && s[i] != '"') {
		const char c = s[i++];
		if (c != '\\') {
			out += c;
			continue;
		}
		if (i >= s.size()) break;
		const char e = s[i++];
		switch (e) {
			case 'n': out += '\n'; break;
			case 't': out += '\t'; break;
			case 'r': out += '\r'; break;
			case 'b': out += '\b'; break;
			case 'f': out += '\f'; break;
			case 'u': {
				if (i + 4 > s.size()) throw std::runtime_error("meta.json: 잘린 \\u 이스케이프");
				unsigned cp = std::stoul(s.substr(i, 4), nullptr, 16);
				i += 4;
				if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
					const unsigned lo = std::stoul(s.substr(i + 2, 4), nullptr, 16);
					if (lo >= 0xDC00 && lo <= 0xDFFF) {
						cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
						i += 6;
					}
				}
				append_utf8(out, cp);
				break;
			}
			default: out += e;  // " \ /
		}
	}
	if (i >= s.size()) throw std::runtime_error("meta.json: 닫히지 않은 문자열");
	++i;
	return out;
}

std::map<std::string, std::string> parse_flat_json(const std::string& text) {
	std::map<std::string, std::string> out;
	size_t i = 0;
	skip_ws(text, i);
	if (i >= text.size() || text[i] != '{') throw std::runtime_error("meta.json: '{'로 시작해야 합니다");
	++i;
	skip_ws(text, i);
	if (i < text.size() && text[i] == '}') return out;

	for (;;) {
		skip_ws(text, i);
		const std::string key = parse_json_string(text, i);
		skip_ws(text, i);
		if (i >= text.size() || text[i] != ':') throw std::runtime_error("meta.json: ':'가 없습니다 (" + key + ")");
		++i;
		skip_ws(text, i);
		if (i >= text.size()) throw std::runtime_error("meta.json: 값이 없습니다 (" + key + ")");

		if (text[i] == '"') {
			out[key] = parse_json_string(text, i);
		} else if (text[i] == '{' || text[i] == '[') {
			throw std::runtime_error("meta.json: 중첩 값은 지원하지 않습니다 (" + key + ")");
		} else {
			const size_t start = i;
			while (i < text.size() && text[i] != ',' && text[i] != '}' && static_cast<unsigned char>(text[i]) > ' ') ++i;
			out[key] = text.substr(start, i - start);
		}

		skip_ws(text, i);
		if (i < text.size() && text[i] == ',') {
			++i;
			continue;
		}
		if (i < text.size() && text[i] == '}') break;
		throw std::runtime_error("meta.json: ',' 또는 '}'가 필요합니다");
	}
	return out;
}

}  // namespace

ModelMeta ModelMeta::load(const fs::path& path) {
	const std::vector<uint8_t> raw = read_file(path);
	std::string text(raw.begin(), raw.end());
	if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF) text.erase(0, 3);  // UTF-8 BOM

	const std::map<std::string, std::string> kv = parse_flat_json(text);
	auto get = [&](const char* key) -> const std::string* {
		auto it = kv.find(key);
		return it == kv.end() ? nullptr : &it->second;
	};
	auto require = [&](const char* key) -> const std::string& {
		const std::string* v = get(key);
		if (v == nullptr) throw std::runtime_error(std::string("meta.json: '") + key + "' 항목이 없습니다");
		return *v;
	};

	ModelMeta m;
	m.captcha_id = require("captcha_id");
	m.image_width = std::stoi(require("image_width"));
	m.image_height = std::stoi(require("image_height"));
	m.label_length = std::stoi(require("label_length"));
	m.characters = require("characters");
	if (const std::string* v = get("threshold")) m.threshold = std::stoi(*v);
	if (const std::string* v = get("preprocess")) m.preprocess = *v;
	return m;
}

// ---------------------------------------------------------------- 인자

namespace {

constexpr size_t kBeamWidth = 10;
constexpr int kIntraThreads = 1;

void usage() {
	std::cerr << "Usage: captcha -c=\"gov24\" -i=\"path/to/image.jpg\""
	             " [-m=\"path/to/model\"] [--meta=\"path/to/meta.json\"]\n";
}

Str unquote(Str s) {
	while (!s.empty() && s.front() == ORT_TSTR('"')) s.erase(s.begin());
	while (!s.empty() && s.back() == ORT_TSTR('"')) s.pop_back();
	return s;
}

// -x=value / --long=value / -x value / --long value 를 모두 받는다 (C# 원본과 동일).
bool match(const std::vector<Str>& args, size_t& i, const Chr* short_form, const Chr* long_form, Str& out) {
	const Str& arg = args[i];
	for (const Chr* form : {short_form, long_form}) {
		const Str prefix = Str(form) + ORT_TSTR("=");
		if (arg.rfind(prefix, 0) == 0) {
			out = unquote(arg.substr(prefix.size()));
			return true;
		}
	}
	if ((arg == short_form || arg == long_form) && i + 1 < args.size()) {
		out = unquote(args[++i]);
		return true;
	}
	return false;
}

int run(const std::vector<Str>& args) {
	Str captcha_id;
	fs::path image_path, model_path, meta_path;

	for (size_t i = 0; i < args.size(); ++i) {
		Str value;
		if (match(args, i, ORT_TSTR("-c"), ORT_TSTR("--captcha-id"), value)) captcha_id = value;
		else if (match(args, i, ORT_TSTR("-i"), ORT_TSTR("--image-path"), value)) image_path = value;
		else if (match(args, i, ORT_TSTR("-m"), ORT_TSTR("--model-path"), value)) model_path = value;
		else if (match(args, i, ORT_TSTR("--meta"), ORT_TSTR("--meta-path"), value)) meta_path = value;
	}

	if (captcha_id.empty()) {
		std::cerr << "Error: captcha-id is required. Use -c or --captcha-id\n";
		usage();
		return 1;
	}
	if (image_path.empty()) {
		std::cerr << "Error: image-path is required. Use -i or --image-path\n";
		usage();
		return 1;
	}

	const fs::path dir = exe_dir();
	if (model_path.empty()) model_path = dir / (captcha_id + ORT_TSTR(".model"));

	if (meta_path.empty()) {
		meta_path = fs::path(model_path).replace_extension(ORT_TSTR(".meta.json"));
		// 모델을 -m으로 직접 준 경우 사이드카가 없을 수 있다. 실행 파일 옆을 한 번 더 본다.
		if (!file_exists(meta_path)) meta_path = dir / (captcha_id + ORT_TSTR(".meta.json"));
	}

	if (!file_exists(model_path)) {
		std::cerr << "Error: model not found: " << narrow(model_path) << "\n";
		return 2;
	}
	if (!file_exists(meta_path)) {
		std::cerr << "Error: metadata not found: " << narrow(meta_path) << "\n"
		          << "Place <model>.meta.json next to the model (see apps/cli/models/).\n";
		return 2;
	}

	const ModelMeta meta = ModelMeta::load(meta_path);
	const std::vector<float> input = preprocess_image(read_file(image_path), meta);

	Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "captcha");
	Ort::SessionOptions options;
	options.SetIntraOpNumThreads(kIntraThreads);
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	Ort::Session session(env, model_path.c_str(), options);

	// 모델이 기대하는 입력 크기와 메타데이터가 다르면 여기서 알아보기 쉽게 끊는다.
	const std::vector<int64_t> model_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	if (model_shape.size() == 4 && model_shape[2] > 0 && model_shape[3] > 0 &&
	    (model_shape[2] != meta.image_height || model_shape[3] != meta.image_width)) {
		std::cerr << "Error: model input " << model_shape[3] << "x" << model_shape[2]
		          << " != metadata " << meta.image_width << "x" << meta.image_height << "\n"
		          << "Fix meta.json or re-export the ONNX at that size.\n";
		return 1;
	}

	Ort::AllocatorWithDefaultOptions allocator;
	const Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, allocator);
	const Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, allocator);
	const char* in_names[] = {in_name.get()};
	const char* out_names[] = {out_name.get()};

	const int64_t shape[4] = {1, 1, meta.image_height, meta.image_width};
	const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value tensor =
	    Ort::Value::CreateTensor<float>(mem, const_cast<float*>(input.data()), input.size(), shape, 4);

	std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

	const std::vector<int64_t> dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	if (dims.size() != 3) {
		std::cerr << "Error: unexpected output rank " << dims.size() << " (expected [T, 1, C])\n";
		return 1;
	}
	const size_t num_frames = static_cast<size_t>(dims[0]);
	const size_t num_classes = static_cast<size_t>(dims[2]);

	const std::vector<std::string> charset = split_utf8(meta.characters);
	if (num_classes != charset.size() + 1) {
		std::cerr << "Error: model classes (" << num_classes << ") != metadata characters ("
		          << charset.size() << " + blank)\n";
		return 1;
	}

	const std::vector<std::vector<double>> log_probs =
	    log_softmax_frames(outputs[0].GetTensorData<float>(), num_frames, num_classes);
	const Decoded decoded =
	    ctc_beam_decode_fixed_length(log_probs, charset, static_cast<size_t>(meta.label_length), kBeamWidth);

	std::fwrite(decoded.text.data(), 1, decoded.text.size(), stdout);
	std::fflush(stdout);
	return 0;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
	try {
		return run(std::vector<Str>(argv + 1, argv + argc));
	} catch (const Ort::Exception& e) {
		std::cerr << "Error: ONNX Runtime: " << e.what() << "\n";
		return 1;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
