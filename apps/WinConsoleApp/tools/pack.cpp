// 빌드 타임 도구. 파일을 LZMS 로 압축해 [uint64 원본 크기][압축 바이트] 로 쓴다.
// captcha.exe 가 이 결과를 RCDATA 리소스로 품고, 실행할 때 풀어서 onnxruntime.dll 을 복원한다.
//
//   pack <in> <out>
#include <windows.h>

#include <compressapi.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int wmain(int argc, wchar_t** argv) {
	if (argc != 3) {
		std::fwprintf(stderr, L"usage: pack <in> <out>\n");
		return 1;
	}

	std::ifstream in(argv[1], std::ios::binary);
	if (!in) {
		std::fwprintf(stderr, L"pack: cannot read %s\n", argv[1]);
		return 1;
	}
	const std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

	COMPRESSOR_HANDLE compressor = nullptr;
	if (!CreateCompressor(COMPRESS_ALGORITHM_LZMS, nullptr, &compressor)) {
		std::fwprintf(stderr, L"pack: CreateCompressor failed (%lu)\n", GetLastError());
		return 1;
	}

	// 크기 조회는 ERROR_INSUFFICIENT_BUFFER 로 실패하며 needed 만 채워진다.
	SIZE_T needed = 0;
	Compress(compressor, raw.data(), raw.size(), nullptr, 0, &needed);
	std::vector<uint8_t> packed(needed);
	SIZE_T packed_size = 0;
	if (!Compress(compressor, raw.data(), raw.size(), packed.data(), packed.size(), &packed_size)) {
		std::fwprintf(stderr, L"pack: Compress failed (%lu)\n", GetLastError());
		CloseCompressor(compressor);
		return 1;
	}
	CloseCompressor(compressor);

	const uint64_t raw_size = raw.size();
	std::ofstream out(argv[2], std::ios::binary);
	out.write(reinterpret_cast<const char*>(&raw_size), sizeof raw_size);
	out.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed_size));
	if (!out) {
		std::fwprintf(stderr, L"pack: cannot write %s\n", argv[2]);
		return 1;
	}

	std::fwprintf(stderr, L"pack: %llu -> %llu bytes\n", raw_size, static_cast<uint64_t>(packed_size));
	return 0;
}
