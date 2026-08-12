// decode.rs의 포팅.
//
// prefix별로 blank로 끝나는 경로(p_b)와 문자로 끝나는 경로(p_nb)의 확률을
// log-sum-exp로 합산한다. beam 점수가 곧 P(문자열 | 이미지)이며,
// 신뢰도는 최종 후보 집합에서 정규화한 사후확률이다.

#include "captcha.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

namespace {

const double NEG_INF = -std::numeric_limits<double>::infinity();

double log_add(double a, double b) {
	if (a == NEG_INF) return b;
	if (b == NEG_INF) return a;
	const double hi = a > b ? a : b;
	const double lo = a > b ? b : a;
	return hi + std::log1p(std::exp(lo - hi));
}

// 프레임 확률 상위 k개 클래스 인덱스 (blank는 항상 포함).
std::vector<size_t> top_classes(const std::vector<double>& frame, size_t k) {
	std::vector<size_t> order(frame.size());
	std::iota(order.begin(), order.end(), size_t{0});
	if (k >= frame.size()) return order;

	// 동점은 인덱스 오름차순으로 갈라 결과를 결정적으로 만든다.
	std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](size_t a, size_t b) {
		if (frame[a] != frame[b]) return frame[a] > frame[b];
		return a < b;
	});
	order.resize(k);
	if (std::find(order.begin(), order.end(), size_t{0}) == order.end()) order.push_back(0);
	return order;
}

using Prefix = std::vector<size_t>;
using Scores = std::array<double, 2>;  // [p_blank, p_nonblank]
using Beams = std::map<Prefix, Scores>;

}  // namespace

std::vector<std::string> split_utf8(const std::string& s) {
	std::vector<std::string> out;
	for (size_t i = 0; i < s.size();) {
		const unsigned char c = static_cast<unsigned char>(s[i]);
		size_t len = 1;
		if (c >= 0xF0) len = 4;
		else if (c >= 0xE0) len = 3;
		else if (c >= 0xC0) len = 2;
		len = std::min(len, s.size() - i);
		out.push_back(s.substr(i, len));
		i += len;
	}
	return out;
}

std::vector<std::vector<double>> log_softmax_frames(const float* logits, size_t num_frames, size_t num_classes) {
	std::vector<std::vector<double>> frames;
	frames.reserve(num_frames);
	for (size_t t = 0; t < num_frames; ++t) {
		const float* row = logits + t * num_classes;
		double max = -std::numeric_limits<double>::infinity();
		for (size_t c = 0; c < num_classes; ++c) max = std::max(max, static_cast<double>(row[c]));
		double sum_exp = 0.0;
		for (size_t c = 0; c < num_classes; ++c) sum_exp += std::exp(static_cast<double>(row[c]) - max);
		const double log_sum = max + std::log(sum_exp);

		std::vector<double> frame(num_classes);
		for (size_t c = 0; c < num_classes; ++c) frame[c] = static_cast<double>(row[c]) - log_sum;
		frames.push_back(std::move(frame));
	}
	return frames;
}

Decoded ctc_beam_decode_fixed_length(const std::vector<std::vector<double>>& log_probs,
                                     const std::vector<std::string>& charset,
                                     size_t expected_length,
                                     size_t beam_width) {
	const size_t num_frames = log_probs.size();
	if (num_frames == 0 || expected_length == 0) return {};

	const size_t num_classes = log_probs[0].size();
	const size_t candidates_per_frame = std::min(beam_width * 2, num_classes);

	Beams beams;
	beams[Prefix{}] = Scores{0.0, NEG_INF};

	for (size_t t = 0; t < num_frames; ++t) {
		const std::vector<double>& frame = log_probs[t];
		const size_t remaining = num_frames - t - 1;
		const std::vector<size_t> classes = top_classes(frame, candidates_per_frame);
		Beams next;

		for (const auto& kv : beams) {
			const Prefix& prefix = kv.first;
			const double p_b = kv.second[0];
			const double p_nb = kv.second[1];
			const double p_total = log_add(p_b, p_nb);
			const bool has_last = !prefix.empty();
			const size_t last = has_last ? prefix.back() : 0;

			// 1) blank: prefix 유지
			{
				auto it = next.emplace(prefix, Scores{NEG_INF, NEG_INF}).first;
				it->second[0] = log_add(it->second[0], p_total + frame[0]);
			}

			for (size_t c : classes) {
				if (c == 0) continue;
				const double lp = frame[c];

				double extended;
				if (has_last && c == last) {
					// 같은 문자 반복: blank 없이 이어지면 같은 prefix
					auto it = next.emplace(prefix, Scores{NEG_INF, NEG_INF}).first;
					it->second[1] = log_add(it->second[1], p_nb + lp);
					extended = p_b + lp;
				} else {
					extended = p_total + lp;
				}

				if (extended == NEG_INF) continue;

				Prefix new_prefix = prefix;
				new_prefix.push_back(c);

				// 고정 길이 제약: 초과 금지 + 남은 프레임으로 도달 불가하면 버림
				if (new_prefix.size() > expected_length) continue;
				if (expected_length - new_prefix.size() > remaining) continue;

				auto it = next.emplace(std::move(new_prefix), Scores{NEG_INF, NEG_INF}).first;
				it->second[1] = log_add(it->second[1], extended);
			}
		}

		Beams filtered;
		for (const auto& kv : next) {
			const size_t need = kv.first.size() >= expected_length ? 0 : expected_length - kv.first.size();
			if (need <= remaining) filtered.insert(kv);
		}
		beams = filtered.empty() ? std::move(next) : std::move(filtered);

		if (beams.size() > beam_width) {
			std::vector<std::pair<Prefix, Scores>> ranked(beams.begin(), beams.end());
			std::partial_sort(ranked.begin(), ranked.begin() + beam_width, ranked.end(),
			                  [](const std::pair<Prefix, Scores>& a, const std::pair<Prefix, Scores>& b) {
				                  const double sa = log_add(a.second[0], a.second[1]);
				                  const double sb = log_add(b.second[0], b.second[1]);
				                  if (sa != sb) return sa > sb;
				                  return a.first < b.first;
			                  });
			ranked.resize(beam_width);
			beams = Beams(ranked.begin(), ranked.end());
		}
	}

	if (beams.empty()) return {};

	std::vector<std::pair<const Prefix*, double>> scored;
	scored.reserve(beams.size());
	for (const auto& kv : beams) scored.emplace_back(&kv.first, log_add(kv.second[0], kv.second[1]));

	// 기대 길이에 맞는 후보 우선, 없으면 전체에서 최고 점수
	const Prefix* best_prefix = nullptr;
	double best_score = NEG_INF;
	for (const auto& s : scored) {
		if (s.first->size() != expected_length) continue;
		if (best_prefix == nullptr || s.second > best_score) {
			best_prefix = s.first;
			best_score = s.second;
		}
	}
	if (best_prefix == nullptr) {
		for (const auto& s : scored) {
			if (best_prefix == nullptr || s.second > best_score) {
				best_prefix = s.first;
				best_score = s.second;
			}
		}
	}

	double total = NEG_INF;
	for (const auto& s : scored) total = log_add(total, s.second);
	double confidence = total > NEG_INF ? std::exp(best_score - total) : 0.0;
	confidence = std::min(1.0, std::max(0.0, confidence));

	Decoded out;
	for (size_t idx : *best_prefix) {
		out.text += (idx >= 1 && idx - 1 < charset.size()) ? charset[idx - 1] : "?";
	}
	out.confidence = confidence;
	return out;
}
