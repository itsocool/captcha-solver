package kr.co.hyperinfo.captchaSolver.ml;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code apps/cli/src/decode.rs} 의 Java 포팅 (원본은 {@code core.py: ctc_beam_decode_fixed_length}).
 *
 * <p>prefix 별로 blank 로 끝나는 경로(p_b)와 문자로 끝나는 경로(p_nb)의 확률을 log-sum-exp 로
 * 합산한다. beam 점수가 곧 P(문자열 | 이미지) 이며, 신뢰도는 최종 후보 집합에서 정규화한 사후확률이다.
 */
public final class CtcDecoder {

	private static final double NEG_INF = Double.NEGATIVE_INFINITY;
	private static final int BLANK = 0;

	private CtcDecoder() {
	}

	public record Decoded(String text, double confidence) {
	}

	private static double logAdd(double a, double b) {
		if (a == NEG_INF) {
			return b;
		}
		if (b == NEG_INF) {
			return a;
		}
		double hi = Math.max(a, b);
		double lo = Math.min(a, b);
		return hi + Math.log1p(Math.exp(lo - hi));
	}

	/** 프레임 확률 상위 {@code k} 개 클래스 인덱스 (blank 는 항상 포함). */
	private static int[] topClasses(double[] frame, int k) {
		if (k >= frame.length) {
			int[] all = new int[frame.length];
			for (int i = 0; i < all.length; i++) {
				all[i] = i;
			}
			return all;
		}

		Integer[] order = new Integer[frame.length];
		for (int i = 0; i < order.length; i++) {
			order[i] = i;
		}
		java.util.Arrays.sort(order, (a, b) -> Double.compare(frame[b], frame[a]));

		boolean hasBlank = false;
		for (int i = 0; i < k; i++) {
			if (order[i] == BLANK) {
				hasBlank = true;
				break;
			}
		}

		int[] selected = new int[hasBlank ? k : k + 1];
		for (int i = 0; i < k; i++) {
			selected[i] = order[i];
		}
		if (!hasBlank) {
			selected[k] = BLANK;
		}
		return selected;
	}

	/**
	 * @param logProbs (T, C) log 확률. 인덱스 0 은 blank.
	 * @return 디코딩된 문자열과 신뢰도
	 */
	public static Decoded decodeFixedLength(double[][] logProbs, char[] charset, int expectedLength, int beamWidth) {
		int numFrames = logProbs.length;
		if (numFrames == 0 || expectedLength == 0) {
			return new Decoded("", 0.0);
		}
		int numClasses = logProbs[0].length;
		int candidatesPerFrame = Math.min(beamWidth * 2, numClasses);

		// prefix(문자 인덱스) -> [p_blank, p_nonblank]
		Map<List<Integer>, double[]> beams = new HashMap<>();
		beams.put(List.of(), new double[] { 0.0, NEG_INF });

		for (int t = 0; t < numFrames; t++) {
			double[] frame = logProbs[t];
			int remaining = numFrames - t - 1;
			int[] classes = topClasses(frame, candidatesPerFrame);
			Map<List<Integer>, double[]> next = new HashMap<>();

			for (Map.Entry<List<Integer>, double[]> beam : beams.entrySet()) {
				List<Integer> prefix = beam.getKey();
				double pBlank = beam.getValue()[0];
				double pNonBlank = beam.getValue()[1];
				double pTotal = logAdd(pBlank, pNonBlank);
				Integer last = prefix.isEmpty() ? null : prefix.get(prefix.size() - 1);

				// 1) blank: prefix 유지
				double[] kept = next.computeIfAbsent(prefix, key -> new double[] { NEG_INF, NEG_INF });
				kept[0] = logAdd(kept[0], pTotal + frame[BLANK]);

				for (int c : classes) {
					if (c == BLANK) {
						continue;
					}
					double lp = frame[c];
					double extended;

					if (last != null && last == c) {
						// 같은 문자 반복: blank 없이 이어지면 같은 prefix
						double[] same = next.computeIfAbsent(prefix, key -> new double[] { NEG_INF, NEG_INF });
						same[1] = logAdd(same[1], pNonBlank + lp);
						extended = pBlank + lp;
					} else {
						extended = pTotal + lp;
					}

					if (extended == NEG_INF) {
						continue;
					}

					// 고정 길이 제약: 초과 금지 + 남은 프레임으로 도달 불가하면 버림
					if (prefix.size() + 1 > expectedLength) {
						continue;
					}
					if (expectedLength - (prefix.size() + 1) > remaining) {
						continue;
					}

					List<Integer> newPrefix = new ArrayList<>(prefix.size() + 1);
					newPrefix.addAll(prefix);
					newPrefix.add(c);

					double[] entry = next.computeIfAbsent(List.copyOf(newPrefix), key -> new double[] { NEG_INF, NEG_INF });
					entry[1] = logAdd(entry[1], extended);
				}
			}

			Map<List<Integer>, double[]> filtered = new HashMap<>();
			for (Map.Entry<List<Integer>, double[]> entry : next.entrySet()) {
				if (Math.max(expectedLength - entry.getKey().size(), 0) <= remaining) {
					filtered.put(entry.getKey(), entry.getValue());
				}
			}
			beams = filtered.isEmpty() ? next : filtered;

			if (beams.size() > beamWidth) {
				List<Map.Entry<List<Integer>, double[]>> ranked = new ArrayList<>(beams.entrySet());
				ranked.sort(Comparator.comparingDouble(
						(Map.Entry<List<Integer>, double[]> e) -> logAdd(e.getValue()[0], e.getValue()[1])).reversed());
				Map<List<Integer>, double[]> trimmed = new HashMap<>();
				for (int i = 0; i < beamWidth; i++) {
					trimmed.put(ranked.get(i).getKey(), ranked.get(i).getValue());
				}
				beams = trimmed;
			}
		}

		if (beams.isEmpty()) {
			return new Decoded("", 0.0);
		}

		List<Integer> bestPrefix = null;
		double bestScore = NEG_INF;
		double bestAnyScore = NEG_INF;
		List<Integer> bestAnyPrefix = null;
		double total = NEG_INF;

		for (Map.Entry<List<Integer>, double[]> entry : beams.entrySet()) {
			double score = logAdd(entry.getValue()[0], entry.getValue()[1]);
			total = logAdd(total, score);

			if (bestAnyPrefix == null || score > bestAnyScore) {
				bestAnyScore = score;
				bestAnyPrefix = entry.getKey();
			}
			// 기대 길이에 맞는 후보 우선
			if (entry.getKey().size() == expectedLength && (bestPrefix == null || score > bestScore)) {
				bestScore = score;
				bestPrefix = entry.getKey();
			}
		}

		if (bestPrefix == null) {
			bestPrefix = bestAnyPrefix;
			bestScore = bestAnyScore;
		}

		double confidence = total > NEG_INF ? Math.exp(bestScore - total) : 0.0;

		StringBuilder text = new StringBuilder(bestPrefix.size());
		for (int index : bestPrefix) {
			text.append(index - 1 < charset.length && index >= 1 ? charset[index - 1] : '?');
		}

		return new Decoded(text.toString(), Math.clamp(confidence, 0.0, 1.0));
	}

	/** (T, N=1, C) 로짓을 (T, C) log 확률로 변환한다. ONNX 출력은 softmax 이전 값이다. */
	public static double[][] logSoftmaxFrames(float[] logits, int numFrames, int numClasses) {
		double[][] frames = new double[numFrames][numClasses];
		for (int t = 0; t < numFrames; t++) {
			int offset = t * numClasses;
			double max = Double.NEGATIVE_INFINITY;
			for (int c = 0; c < numClasses; c++) {
				max = Math.max(max, logits[offset + c]);
			}
			double sumExp = 0.0;
			for (int c = 0; c < numClasses; c++) {
				sumExp += Math.exp(logits[offset + c] - max);
			}
			double logSum = max + Math.log(sumExp);
			for (int c = 0; c < numClasses; c++) {
				frames[t][c] = logits[offset + c] - logSum;
			}
		}
		return frames;
	}
}
