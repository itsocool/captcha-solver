// decode.rs의 단위 테스트 포팅. 프레임워크 없이 assert만 쓴다.
#undef NDEBUG  // Release에서도 assert가 살아 있어야 한다
#include "captcha.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

const std::vector<std::string> kCharset = {"a", "b", "c"};

// 확률 행(합 1.0) -> log 확률 프레임
std::vector<std::vector<double>> frames(const std::vector<std::vector<double>>& rows) {
	std::vector<std::vector<double>> out;
	for (const auto& row : rows) {
		double sum = 0.0;
		for (double p : row) sum += p;
		assert(std::fabs(sum - 1.0) < 1e-9 && "각 프레임 확률 합이 1이어야 합니다");
		std::vector<double> lp;
		for (double p : row) lp.push_back(std::log(p > 1e-12 ? p : 1e-12));
		out.push_back(std::move(lp));
	}
	return out;
}

void confident_prediction() {
	const auto lp = frames({
	    {0.01, 0.97, 0.01, 0.01},  // a
	    {0.97, 0.01, 0.01, 0.01},  // blank
	    {0.01, 0.01, 0.97, 0.01},  // b
	});
	const Decoded d = ctc_beam_decode_fixed_length(lp, kCharset, 2, 10);
	assert(d.text == "ab");
	assert(d.confidence > 0.95);
}

void tie_gives_half_confidence() {
	const auto lp = frames({{0.0, 0.5, 0.5, 0.0}, {1.0, 0.0, 0.0, 0.0}});
	const Decoded d = ctc_beam_decode_fixed_length(lp, kCharset, 1, 10);
	assert(d.text == "a" || d.text == "b");
	assert(std::fabs(d.confidence - 0.5) < 1e-6);
}

void sums_over_alignments() {
	// 'a': (a,a) + (a,blank) = 0.5 / 'b': (b,blank) = 0.25 -> 2/3
	// 경로 최댓값만 쓰면 0.5가 나온다.
	const auto lp = frames({{0.0, 0.5, 0.5, 0.0}, {0.5, 0.5, 0.0, 0.0}});
	const Decoded d = ctc_beam_decode_fixed_length(lp, kCharset, 1, 10);
	assert(d.text == "a");
	assert(std::fabs(d.confidence - 2.0 / 3.0) < 1e-6);
}

void repeated_characters_need_blank() {
	const auto lp = frames({
	    {0.01, 0.97, 0.01, 0.01},
	    {0.97, 0.01, 0.01, 0.01},
	    {0.01, 0.97, 0.01, 0.01},
	});
	const Decoded d = ctc_beam_decode_fixed_length(lp, kCharset, 2, 10);
	assert(d.text == "aa");
	assert(d.confidence > 0.9);
}

void fixed_length_is_enforced() {
	const auto lp = frames(std::vector<std::vector<double>>(6, {0.7, 0.1, 0.1, 0.1}));
	const Decoded d = ctc_beam_decode_fixed_length(lp, kCharset, 3, 10);
	assert(d.text.size() == 3);
	assert(d.confidence >= 0.0 && d.confidence <= 1.0);
}

void log_softmax_normalizes() {
	const float logits[] = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f};
	for (const auto& frame : log_softmax_frames(logits, 2, 3)) {
		double total = 0.0;
		for (double v : frame) total += std::exp(v);
		assert(std::fabs(total - 1.0) < 1e-9);
	}
}

void utf8_split() {
	const auto cs = split_utf8("0aA가");
	assert(cs.size() == 4);
	assert(cs[3] == "가");
}

}  // namespace

int main() {
	confident_prediction();
	tie_gives_half_confidence();
	sums_over_alignments();
	repeated_characters_need_blank();
	fixed_length_is_enforced();
	log_softmax_normalizes();
	utf8_split();
	std::puts("ok");
	return 0;
}
