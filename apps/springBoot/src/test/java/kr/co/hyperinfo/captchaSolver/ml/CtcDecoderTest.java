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
