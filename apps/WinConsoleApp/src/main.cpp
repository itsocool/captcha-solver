// apps/cli(Rust)의 포팅. 인자 문법과 출력 형식을 그대로 맞췄다.
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

#ifdef CAPTCHA_SINGLE_EXE
// API 포인터를 헤더가 알아서 채우지 않는다. load_embedded_ort() 가 직접 넣는다.
#define ORT_API_MANUAL_INIT
#include <compressapi.h>
#endif

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
	// 항상 UTF-8. 소스가 /utf-8 이라 메시지 리터럴도 UTF-8 이고, wmain 이 콘솔 출력
	// 코드페이지를 UTF-8 로 맞춰 둔다. 리다이렉트된 출력도 이 인코딩으로 일관된다.
	const std::wstring& s = p.native();
	const int n = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
	std::string out(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n, nullptr, nullptr);
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

#ifdef CAPTCHA_SINGLE_EXE

// 같은 DLL 을 매번 풀지 않도록 크기로 구분한 캐시 디렉터리를 쓴다.
fs::path ort_cache_dir(uint64_t size) {
	fs::path base;
	if (const wchar_t* local = _wgetenv(L"LOCALAPPDATA")) {
		base = local;
	} else {
		std::wstring buf(MAX_PATH + 1, L'\0');
		buf.resize(GetTempPathW(static_cast<DWORD>(buf.size()), buf.data()));
		base = buf;
	}
	return base / L"captcha-solver" / (L"ort-" + std::to_wstring(size));
}

// 실행 파일에 넣어둔 onnxruntime.dll 을 캐시에 풀고 로드한다.
// ponytail: 캐시는 사용자 쓰기 가능 경로다. 같은 계정의 다른 프로세스가 DLL 을 바꿔칠 수 있지만
//           그 권한이면 exe 자체를 고칠 수 있으므로 막지 않는다. 필요해지면 서명 검증을 넣을 것.
void load_embedded_ort() {
	const HMODULE self = GetModuleHandleW(nullptr);
	// RT_RCDATA 는 UNICODE 매크로에 따라 A/W 가 갈리므로 값(10)을 직접 쓴다.
	const HRSRC info = FindResourceW(self, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(10));
	const HGLOBAL res = info != nullptr ? LoadResource(self, info) : nullptr;
	const auto* blob = res != nullptr ? static_cast<const uint8_t*>(LockResource(res)) : nullptr;
	const DWORD blob_size = info != nullptr ? SizeofResource(self, info) : 0;
	if (blob == nullptr || blob_size <= sizeof(uint64_t)) throw std::runtime_error("내장된 onnxruntime 을 찾을 수 없습니다");

	uint64_t raw_size = 0;
	std::memcpy(&raw_size, blob, sizeof raw_size);

	const fs::path dll = ort_cache_dir(raw_size) / L"onnxruntime.dll";
	std::error_code ec;
	if (fs::file_size(dll, ec) != raw_size) {
		std::vector<uint8_t> raw(static_cast<size_t>(raw_size));
		DECOMPRESSOR_HANDLE decompressor = nullptr;
		if (!CreateDecompressor(COMPRESS_ALGORITHM_LZMS, nullptr, &decompressor))
			throw std::runtime_error("CreateDecompressor 실패");
		SIZE_T n = 0;
		const BOOL ok = Decompress(decompressor, blob + sizeof raw_size, blob_size - sizeof raw_size, raw.data(),
		                           raw.size(), &n);
		CloseDecompressor(decompressor);
		if (!ok || n != raw.size()) throw std::runtime_error("onnxruntime 압축 해제 실패");

		fs::create_directories(dll.parent_path(), ec);
		// 두 프로세스가 같은 캐시를 동시에 만들 수 있다. 임시 파일에 쓰고 rename 한다.
		const fs::path tmp = dll.parent_path() / (L"onnxruntime.dll." + std::to_wstring(GetCurrentProcessId()));
		{
			std::ofstream out(tmp, std::ios::binary);
			if (!out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
				throw std::runtime_error("쓸 수 없습니다: " + narrow(tmp));
		}
		// 다른 프로세스가 먼저 만들어 로드 중이면 rename 이 실패한다. 그때는 그 파일을 그대로 쓴다.
		if (!MoveFileExW(tmp.c_str(), dll.c_str(), MOVEFILE_REPLACE_EXISTING)) fs::remove(tmp, ec);
	}

	const HMODULE handle = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	if (handle == nullptr) throw std::runtime_error("로드할 수 없습니다: " + narrow(dll));
	using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
	const auto get_api_base = reinterpret_cast<GetApiBaseFn>(GetProcAddress(handle, "OrtGetApiBase"));
	if (get_api_base == nullptr) throw std::runtime_error("OrtGetApiBase 가 없습니다");
	Ort::InitApi(get_api_base()->GetApi(ORT_API_VERSION));
}

#endif  // CAPTCHA_SINGLE_EXE

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
	std::cerr << "Usage: captcha -i=\"path/to/image.jpg\" [-c=\"gov24\"]"
	             " [-m=\"path/to/model.ort\"] [--meta=\"path/to/meta.json\"]\n"
	             "       (-c 기본값: supreme_court)\n"
	             "       captcha --to-ort <in.onnx> <out.ort>\n";
}

// ONNX 를 ORT 포맷으로 굽는다. 최적화가 파일에 박혀 세션 여는 시간이 줄어든다.
// 빌드가 모델을 배치할 때 이 모드를 쓰고, 사용자도 모델을 직접 변환할 수 있다.
//
// 최적화는 EXTENDED 까지만 건다. ORT_ENABLE_ALL 은 레이아웃 최적화(NCHWc)를 포함해
// **변환한 기계의 CPU 명령셋에 맞춰 굳으므로**, 다른 CPU 로 옮길 파일에는 쓸 수 없다.
int to_ort(const fs::path& src, const fs::path& dst) {
#ifdef CAPTCHA_SINGLE_EXE
	load_embedded_ort();
#endif
	Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "captcha");
	Ort::SessionOptions options;
	options.SetIntraOpNumThreads(kIntraThreads);
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	options.SetOptimizedModelFilePath(dst.c_str());
	options.AddConfigEntry("session.save_model_format", "ORT");

	// 세션을 여는 순간 최적화된 그래프가 dst 로 떨어진다.
	const Ort::Session session(env, src.c_str(), options);
	return 0;
}

Str unquote(Str s) {
	while (!s.empty() && s.front() == ORT_TSTR('"')) s.erase(s.begin());
	while (!s.empty() && s.back() == ORT_TSTR('"')) s.pop_back();
	return s;
}

// -x=value / --long=value / -x value / --long value 를 모두 받는다.
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

	if (!args.empty() && args[0] == ORT_TSTR("--to-ort")) {
		if (args.size() != 3) {
			usage();
			return 1;
		}
		return to_ort(unquote(args[1]), unquote(args[2]));
	}

	for (size_t i = 0; i < args.size(); ++i) {
		Str value;
		if (match(args, i, ORT_TSTR("-c"), ORT_TSTR("--captcha-id"), value)) captcha_id = value;
		else if (match(args, i, ORT_TSTR("-i"), ORT_TSTR("--image-path"), value)) image_path = value;
		else if (match(args, i, ORT_TSTR("-m"), ORT_TSTR("--model-path"), value)) model_path = value;
		else if (match(args, i, ORT_TSTR("--meta"), ORT_TSTR("--meta-path"), value)) meta_path = value;
	}

	// 캡차를 지정하지 않으면 supreme_court. 파이썬 engine 쪽 기본값과 같다.
	if (captcha_id.empty()) captcha_id = ORT_TSTR("supreme_court");

	if (image_path.empty()) {
		std::cerr << "Error: image-path is required. Use -i or --image-path\n";
		usage();
		return 1;
	}

	const fs::path dir = exe_dir();
	// 기본은 ORT 포맷. ORT 는 확장자로 포맷을 가리므로 .onnx 를 -m 으로 줘도 그대로 열린다.
	if (model_path.empty()) model_path = dir / (captcha_id + ORT_TSTR(".ort"));

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

#ifdef CAPTCHA_SINGLE_EXE
	load_embedded_ort();  // 첫 Ort:: 호출 전에 반드시.
#endif

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
#ifdef _WIN32
	// 메시지 리터럴이 UTF-8(/utf-8)이라 CP949 콘솔에서는 한글이 깨진다. 출력 코드페이지를
	// UTF-8 로 올린다. 콘솔 설정은 프로세스가 끝나도 남으므로 반드시 되돌린다.
	struct ConsoleUtf8 {
		const UINT prev = GetConsoleOutputCP();
		ConsoleUtf8() { if (prev != CP_UTF8) SetConsoleOutputCP(CP_UTF8); }
		~ConsoleUtf8() { if (prev != CP_UTF8 && prev != 0) SetConsoleOutputCP(prev); }
	} console_utf8;
#endif

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
