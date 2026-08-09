package kr.co.hyperinfo.captchaSolver.ml;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** apps/cli/src/decode.rs 의 테스트를 그대로 옮겨 두 구현이 같은 답을 내는지 고정한다. */
class CtcDecoderTest {

	private static final char[] CHARSET = { 'a', 'b', 'c' };

	/** 각 행은 확률(합=1). log 로 바꿔 넘긴다. */
	private static double[][] frames(double[][] rows) {
		double[][] out = new double[rows.length][];
		for (int i = 0; i < rows.length; i++) {
			double sum = 0;
			for (double p : rows[i]) {
				sum += p;
			}
			assertTrue(Math.abs(sum - 1.0) < 1e-9, "각 프레임 확률 합이 1이어야 합니다");

			out[i] = new double[rows[i].length];
			for (int c = 0; c < rows[i].length; c++) {
				out[i][c] = Math.log(Math.max(rows[i][c], 1e-12));
			}
		}
		return out;
	}

	@Test
	void confidentPrediction() {
		double[][] lp = frames(new double[][] {
				{ 0.01, 0.97, 0.01, 0.01 }, // a
				{ 0.97, 0.01, 0.01, 0.01 }, // blank
				{ 0.01, 0.01, 0.97, 0.01 }, // b
		});
		var decoded = CtcDecoder.decodeFixedLength(lp, CHARSET, 2, 10);
		assertEquals("ab", decoded.text());
		assertTrue(decoded.confidence() > 0.95, "conf = " + decoded.confidence());
	}

	@Test
	void tieGivesHalfConfidence() {
		double[][] lp = frames(new double[][] { { 0.0, 0.5, 0.5, 0.0 }, { 1.0, 0.0, 0.0, 0.0 } });
		var decoded = CtcDecoder.decodeFixedLength(lp, CHARSET, 1, 10);
		assertTrue("a".equals(decoded.text()) || "b".equals(decoded.text()), "text = " + decoded.text());
		assertEquals(0.5, decoded.confidence(), 1e-6);
	}

	@Test
	void sumsOverAlignments() {
		// 'a': (a,a) + (a,blank) = 0.5 / 'b': (b,blank) = 0.25 -> 2/3
		// 경로 최댓값만 쓰면 0.5가 나온다.
		double[][] lp = frames(new double[][] { { 0.0, 0.5, 0.5, 0.0 }, { 0.5, 0.5, 0.0, 0.0 } });
		var decoded = CtcDecoder.decodeFixedLength(lp, CHARSET, 1, 10);
		assertEquals("a", decoded.text());
		assertEquals(2.0 / 3.0, decoded.confidence(), 1e-6);
	}

	@Test
	void repeatedCharactersNeedBlank() {
		double[][] lp = frames(new double[][] {
				{ 0.01, 0.97, 0.01, 0.01 },
				{ 0.97, 0.01, 0.01, 0.01 },
				{ 0.01, 0.97, 0.01, 0.01 },
		});
		var decoded = CtcDecoder.decodeFixedLength(lp, CHARSET, 2, 10);
		assertEquals("aa", decoded.text());
		assertTrue(decoded.confidence() > 0.9, "conf = " + decoded.confidence());
	}

	@Test
	void fixedLengthIsEnforced() {
		double[][] rows = new double[6][];
		for (int i = 0; i < 6; i++) {
			rows[i] = new double[] { 0.7, 0.1, 0.1, 0.1 };
		}
		var decoded = CtcDecoder.decodeFixedLength(frames(rows), CHARSET, 3, 10);
		assertEquals(3, decoded.text().length());
		assertTrue(decoded.confidence() >= 0.0 && decoded.confidence() <= 1.0);
	}

	/** 모든 프레임 경로를 열거해 길이 L 인 라벨열의 확률 합 (작은 입력 전용). */
	private static double bruteForceLengthProb(double[][] logProbs, int expectedLength) {
		int numFrames = logProbs.length;
		int numClasses = logProbs[0].length;
		int combinations = (int) Math.pow(numClasses, numFrames);
		double total = 0.0;
		for (int code = 0; code < combinations; code++) {
			int[] path = new int[numFrames];
			int rest = code;
			for (int t = 0; t < numFrames; t++) {
				path[t] = rest % numClasses;
				rest /= numClasses;
			}
			int collapsed = 0;
			int prev = -1;
			for (int c : path) {
				if (c != prev && c != 0) {
					collapsed++;
				}
				prev = c;
			}
			if (collapsed == expectedLength) {
				double p = 1.0;
				for (int t = 0; t < numFrames; t++) {
					p *= Math.exp(logProbs[t][path[t]]);
				}
				total += p;
			}
		}
		return total;
	}

	private static double[][] mixedFrames() {
		return frames(new double[][] {
				{ 0.4, 0.2, 0.3, 0.1 },
				{ 0.1, 0.5, 0.2, 0.2 },
				{ 0.25, 0.25, 0.25, 0.25 },
				{ 0.6, 0.1, 0.2, 0.1 },
		});
	}

	@Test
	void lengthLogProbMatchesBruteForce() {
		double[][] lp = mixedFrames();
		for (int l = 1; l <= 3; l++) {
			assertEquals(bruteForceLengthProb(lp, l), Math.exp(CtcDecoder.lengthLogProb(lp, l)), 1e-12,
					"L = " + l);
		}
	}

	@Test
	void lengthDistributionSumsToOne() {
		double[][] lp = mixedFrames();
		double total = bruteForceLengthProb(lp, 0);
		for (int l = 1; l <= 4; l++) {
			total += Math.exp(CtcDecoder.lengthLogProb(lp, l));
		}
		assertEquals(1.0, total, 1e-12);
	}

	@Test
	void ambiguousInputIsNotConfident() {
		// 어느 문자도 뚜렷하지 않으면 신뢰도가 낮아야 한다. beam 내부 정규화는
		// 1위와 2위의 비율만 보므로 이런 입력에도 높은 값을 줄 수 있다.
		double[][] rows = new double[6][];
		for (int i = 0; i < 6; i++) {
			rows[i] = new double[] { 0.4, 0.2, 0.2, 0.2 };
		}
		var decoded = CtcDecoder.decodeFixedLength(frames(rows), CHARSET, 3, 10);
		assertEquals(3, decoded.text().length());
		assertTrue(decoded.confidence() < 0.1, "conf = " + decoded.confidence());
	}

	@Test
	void confidenceConvergesAsBeamWidens() {
		// 분모가 beam 과 무관하게 고정이라 값이 단조 증가하며 고정점에 수렴한다.
		double[][] lp = frames(new double[][] {
				{ 0.3, 0.3, 0.2, 0.2 },
				{ 0.2, 0.4, 0.2, 0.2 },
				{ 0.5, 0.2, 0.2, 0.1 },
				{ 0.1, 0.3, 0.4, 0.2 },
				{ 0.4, 0.2, 0.2, 0.2 },
				{ 0.2, 0.2, 0.3, 0.3 },
		});
		int[] widths = { 5, 15, 40, 80 };
		double[] confs = new double[widths.length];
		for (int i = 0; i < widths.length; i++) {
			confs[i] = CtcDecoder.decodeFixedLength(lp, CHARSET, 3, widths[i]).confidence();
		}
		for (int i = 1; i < confs.length; i++) {
			assertTrue(confs[i - 1] <= confs[i] + 1e-12,
					"단조 증가가 깨졌다: " + java.util.Arrays.toString(confs));
		}
		assertEquals(confs[confs.length - 2], confs[confs.length - 1], 1e-9,
				"수렴하지 않았다: " + java.util.Arrays.toString(confs));
	}

	@Test
	void logSoftmaxNormalizes() {
		float[] logits = { 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f };
		for (double[] frame : CtcDecoder.logSoftmaxFrames(logits, 2, 3)) {
			double total = 0;
			for (double v : frame) {
				total += Math.exp(v);
			}
			assertEquals(1.0, total, 1e-9);
		}
	}
}
