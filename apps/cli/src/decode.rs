//! `core.py: ctc_beam_decode_fixed_length`의 Rust 포팅.
//!
//! prefix별로 blank로 끝나는 경로(p_b)와 문자로 끝나는 경로(p_nb)의 확률을
//! log-sum-exp로 합산한다. beam 점수가 곧 P(문자열 | 이미지)이며,
//! 신뢰도는 길이 제약으로 조건부화한 사후확률
//! P(예측 문자열 | 이미지, 길이 = expected_length) 이다.

use std::collections::HashMap;

const NEG_INF: f64 = f64::NEG_INFINITY;

fn log_add(a: f64, b: f64) -> f64 {
	if a == NEG_INF {
		return b;
	}
	if b == NEG_INF {
		return a;
	}
	let (hi, lo) = if a > b { (a, b) } else { (b, a) };
	hi + (lo - hi).exp().ln_1p()
}

/// 프레임 확률 상위 `k`개 클래스 인덱스 (blank는 항상 포함).
fn top_classes(frame: &[f64], k: usize) -> Vec<usize> {
	if k >= frame.len() {
		return (0..frame.len()).collect();
	}
	let mut order: Vec<usize> = (0..frame.len()).collect();
	order.sort_unstable_by(|&a, &b| frame[b].partial_cmp(&frame[a]).unwrap_or(std::cmp::Ordering::Equal));
	let mut selected: Vec<usize> = order.into_iter().take(k).collect();
	if !selected.contains(&0) {
		selected.push(0);
	}
	selected
}

/// log P(|y| = expected_length | 이미지). 길이가 정확히 L인 **모든** 라벨열의 확률 합.
///
/// 신뢰도를 정규화할 때 쓰는 분모다. beam에 무엇이 살아남았는지와 무관하게 결정되므로
/// beam_width를 바꿔도 값이 흔들리지 않는다.
///
/// 상태는 (k = 지금까지 emit한 문자 수, c = 마지막 emit 문자, 0은 "아직 없음"):
///   `b[k][c]` 현재 프레임이 blank / `a[k][c]` 현재 프레임이 문자 c를 emit (c >= 1).
/// 프레임마다 전체 합으로 스케일링해 언더플로를 피하고 로그 스케일만 누적한다.
/// 비용은 O(T * L * C)로 beam search에 비하면 무시할 수준이다.
pub fn length_logprob(log_probs: &[Vec<f64>], expected_length: usize) -> f64 {
	let num_frames = log_probs.len();
	let l = expected_length;
	if l == 0 || num_frames == 0 || num_frames < l {
		return NEG_INF;
	}
	let num_classes = log_probs[0].len();
	let idx = |k: usize, c: usize| k * num_classes + c;

	let mut a = vec![0.0f64; (l + 1) * num_classes];
	let mut b = vec![0.0f64; (l + 1) * num_classes];
	b[idx(0, 0)] = 1.0;
	let mut log_scale = 0.0f64;

	for frame_lp in log_probs {
		let frame: Vec<f64> = frame_lp.iter().map(|v| v.exp()).collect();
		let mut next_a = vec![0.0f64; a.len()];
		let mut next_b = vec![0.0f64; b.len()];

		for k in 0..=l {
			let mut row_sum = 0.0;
			for c in 0..num_classes {
				row_sum += a[idx(k, c)] + b[idx(k, c)];
			}

			// blank: k와 마지막 문자를 그대로 유지
			for c in 0..num_classes {
				next_b[idx(k, c)] = (a[idx(k, c)] + b[idx(k, c)]) * frame[0];
			}

			// 같은 문자를 blank 없이 반복: 문자열이 그대로이므로 k 유지
			for c in 1..num_classes {
				next_a[idx(k, c)] += a[idx(k, c)] * frame[c];
			}

			// 새 문자 c로 확장하여 k+1: 마지막 문자가 c가 아닌 모든 경로와
			// 마지막 문자가 c였지만 blank를 거친 경로의 합 = row_sum - a[k][c]
			if k < l {
				for c in 1..num_classes {
					let mass = (row_sum - a[idx(k, c)]).max(0.0);
					next_a[idx(k + 1, c)] += mass * frame[c];
				}
			}
		}

		a = next_a;
		b = next_b;

		let scale: f64 = a.iter().sum::<f64>() + b.iter().sum::<f64>();
		if scale <= 0.0 {
			return NEG_INF;
		}
		for v in a.iter_mut() {
			*v /= scale;
		}
		for v in b.iter_mut() {
			*v /= scale;
		}
		log_scale += scale.ln();
	}

	let mut tail = 0.0;
	for c in 0..num_classes {
		tail += a[idx(l, c)] + b[idx(l, c)];
	}
	if tail > 0.0 {
		tail.ln() + log_scale
	} else {
		NEG_INF
	}
}

/// `log_probs`: (T, C) log 확률. 인덱스 0은 blank.
/// 반환: (디코딩된 문자열, 신뢰도)
pub fn ctc_beam_decode_fixed_length(
	log_probs: &[Vec<f64>],
	charset: &[char],
	expected_length: usize,
	beam_width: usize,
) -> (String, f64) {
	let num_frames = log_probs.len();
	if num_frames == 0 || expected_length == 0 {
		return (String::new(), 0.0);
	}
	let num_classes = log_probs[0].len();
	let candidates_per_frame = (beam_width * 2).min(num_classes);

	// prefix(문자 인덱스) -> [p_blank, p_nonblank]
	let mut beams: HashMap<Vec<usize>, [f64; 2]> = HashMap::new();
	beams.insert(Vec::new(), [0.0, NEG_INF]);

	for (t, frame) in log_probs.iter().enumerate() {
		let remaining = num_frames - t - 1;
		let classes = top_classes(frame, candidates_per_frame);
		let mut next: HashMap<Vec<usize>, [f64; 2]> = HashMap::new();

		for (prefix, scores) in &beams {
			let (p_b, p_nb) = (scores[0], scores[1]);
			let p_total = log_add(p_b, p_nb);
			let last = prefix.last().copied();

			// 1) blank: prefix 유지
			let entry = next.entry(prefix.clone()).or_insert([NEG_INF, NEG_INF]);
			entry[0] = log_add(entry[0], p_total + frame[0]);

			for &c in &classes {
				if c == 0 {
					continue;
				}
				let lp = frame[c];

				let extended = if Some(c) == last {
					// 같은 문자 반복: blank 없이 이어지면 같은 prefix
					let entry = next.entry(prefix.clone()).or_insert([NEG_INF, NEG_INF]);
					entry[1] = log_add(entry[1], p_nb + lp);
					p_b + lp
				} else {
					p_total + lp
				};

				if extended == NEG_INF {
					continue;
				}

				let mut new_prefix = prefix.clone();
				new_prefix.push(c);

				// 고정 길이 제약: 초과 금지 + 남은 프레임으로 도달 불가하면 버림
				if new_prefix.len() > expected_length {
					continue;
				}
				if expected_length - new_prefix.len() > remaining {
					continue;
				}

				let entry = next.entry(new_prefix).or_insert([NEG_INF, NEG_INF]);
				entry[1] = log_add(entry[1], extended);
			}
		}

		let filtered: HashMap<Vec<usize>, [f64; 2]> = next
			.iter()
			.filter(|(prefix, _)| expected_length.saturating_sub(prefix.len()) <= remaining)
			.map(|(prefix, scores)| (prefix.clone(), *scores))
			.collect();
		beams = if filtered.is_empty() { next } else { filtered };

		if beams.len() > beam_width {
			let mut ranked: Vec<(Vec<usize>, [f64; 2])> = beams.into_iter().collect();
			ranked.sort_by(|a, b| {
				let sa = log_add(a.1[0], a.1[1]);
				let sb = log_add(b.1[0], b.1[1]);
				sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
			});
			ranked.truncate(beam_width);
			beams = ranked.into_iter().collect();
		}
	}

	if beams.is_empty() {
		return (String::new(), 0.0);
	}

	let scored: Vec<(&Vec<usize>, f64)> = beams
		.iter()
		.map(|(prefix, scores)| (prefix, log_add(scores[0], scores[1])))
		.collect();

	// 기대 길이에 맞는 후보 우선, 없으면 전체에서 최고 점수
	let exact: Vec<&(&Vec<usize>, f64)> = scored.iter().filter(|(p, _)| p.len() == expected_length).collect();
	let has_exact = !exact.is_empty();
	let (best_prefix, best_score) = if has_exact {
		let best = exact
			.iter()
			.max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
			.unwrap();
		(best.0, best.1)
	} else {
		let best = scored
			.iter()
			.max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
			.unwrap();
		(best.0, best.1)
	};

	// 신뢰도: 길이 L로 조건부화한 사후확률. 분모는 beam과 무관하게 정확히 구한다.
	// 길이 L 도달 불가(T < L 등)면 남은 후보들 사이의 상대 점수로 대체한다.
	let mut log_z = if has_exact { length_logprob(log_probs, expected_length) } else { NEG_INF };
	if log_z == NEG_INF {
		log_z = scored.iter().fold(NEG_INF, |acc, (_, score)| log_add(acc, *score));
	}
	let confidence = if log_z > NEG_INF { (best_score - log_z).exp() } else { 0.0 };

	let text: String = best_prefix
		.iter()
		.map(|&idx| charset.get(idx - 1).copied().unwrap_or('?'))
		.collect();

	(text, confidence.clamp(0.0, 1.0))
}

/// (T, N=1, C) 로짓을 (T, C) log 확률로 변환한다. ONNX 출력은 softmax 이전 값이다.
pub fn log_softmax_frames(logits: &[f32], num_frames: usize, num_classes: usize) -> Vec<Vec<f64>> {
	let mut frames = Vec::with_capacity(num_frames);
	for t in 0..num_frames {
		let row = &logits[t * num_classes..(t + 1) * num_classes];
		let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
		let sum_exp: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
		let log_sum = max + sum_exp.ln();
		frames.push(row.iter().map(|&v| v as f64 - log_sum).collect());
	}
	frames
}
