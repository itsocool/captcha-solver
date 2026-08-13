import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.stream.Stream;

/**
 * 저장소 자산을 pom.xml 과 같은 위치(models/, db/)로 동기화한다.
 *
 * - models/: captcha_data/&lt;id&gt;/&lt;rev&gt;/model/ 의 최신(가장 큰 rev 번호) model.ort ·
 *   model.meta.json 을 &lt;id&gt;.ort · &lt;id&gt;.meta.json 으로 이름 바꿔 복사한다.
 *   apps/cli/tools/sync_models.py 의 Java 버전. 그쪽은 "현재 리비전"을 hypercaptcha.engine 의
 *   레지스트리(하드코딩)로 정하지만, 여기서는 Python 의존 없이 rev 디렉터리 이름의 최댓값을 쓴다.
 *   즉 학습만 하고 아직 승격 안 한 새 리비전이 더 큰 번호로 있으면 그걸 집어올 수 있다 —
 *   알고 쓰는 트레이드오프.
 * - db/: 저장소 루트 db/ 를 그대로(파일명 유지, 하위 디렉터리 포함) 복사한다.
 *
 * 두 경우 다 출력 디렉터리가 없으면 만들고, 파일이 이미 있으면 크기·수정시각이 같은 한 다시
 * 복사하지 않는다 — generate-resources 단계에서 매번 돌아도 수 MB 짜리 모델을 반복 복사하지
 * 않게 하기 위해서다.
 *
 * 실행: java SyncResource.java <captcha_data 디렉터리> <출력 models 디렉터리> <db 디렉터리> <출력 db 디렉터리>
 */
public class SyncResource {

	public static void main(String[] args) throws IOException {
		if (args.length != 4) {
			System.err.println("usage: java SyncResource.java <captcha_data_dir> <output_models_dir>"
					+ " <db_dir> <output_db_dir>");
			System.exit(2);
		}
		syncModels(Path.of(args[0]), Path.of(args[1]));
		syncDb(Path.of(args[2]), Path.of(args[3]));
	}

	// --- models -------------------------------------------------------------

	private static void syncModels(Path captchaDataDir, Path modelsDir) throws IOException {
		if (!Files.isDirectory(captchaDataDir)) {
			System.out.println("[sync-resource] captcha_data 없음, 건너뜀: " + captchaDataDir);
			return;
		}
		Files.createDirectories(modelsDir);

		try (Stream<Path> ids = Files.list(captchaDataDir)) {
			for (Path idDir : ids.filter(Files::isDirectory).sorted().toList()) {
				syncOneModel(idDir, modelsDir);
			}
		}
	}

	private static void syncOneModel(Path idDir, Path modelsDir) throws IOException {
		String captchaId = idDir.getFileName().toString();
		Path latestRevDir = latestRevisionDir(idDir);
		if (latestRevDir == null) {
			System.out.println("[sync-resource] " + captchaId + ": 리비전 없음, 건너뜀");
			return;
		}

		Path modelDir = latestRevDir.resolve("model");
		Path ort = modelDir.resolve("model.ort");
		Path meta = modelDir.resolve("model.meta.json");
		if (!Files.exists(ort) || !Files.exists(meta)) {
			System.out.println("[sync-resource] " + captchaId + " (rev " + latestRevDir.getFileName()
					+ "): model.ort/model.meta.json 없음, 건너뜀");
			return;
		}

		boolean ortCopied = copyIfChanged(ort, modelsDir.resolve(captchaId + ".ort"));
		boolean metaCopied = copyIfChanged(meta, modelsDir.resolve(captchaId + ".meta.json"));
		System.out.println("[sync-resource] " + captchaId + ": rev " + latestRevDir.getFileName()
				+ (ortCopied || metaCopied ? " 복사 완료" : " 변경 없음, 건너뜀"));
	}

	/** <id>/ 밑의 숫자 이름 서브디렉토리 중 값이 가장 큰 것. */
	private static Path latestRevisionDir(Path idDir) throws IOException {
		try (Stream<Path> revs = Files.list(idDir)) {
			return revs.filter(Files::isDirectory)
					.filter(p -> p.getFileName().toString().matches("\\d+"))
					.max(Comparator.comparingInt(p -> Integer.parseInt(p.getFileName().toString())))
					.orElse(null);
		}
	}

	// --- db -------------------------------------------------------------------

	private static void syncDb(Path dbDir, Path outDbDir) throws IOException {
		if (!Files.isDirectory(dbDir)) {
			System.out.println("[sync-resource] db 없음, 건너뜀: " + dbDir);
			return;
		}
		Files.createDirectories(outDbDir);

		try (Stream<Path> files = Files.walk(dbDir)) {
			for (Path source : files.filter(Files::isRegularFile).sorted().toList()) {
				Path relative = dbDir.relativize(source);
				Path dest = outDbDir.resolve(relative);
				Files.createDirectories(dest.getParent());
				boolean copied = copyIfChanged(source, dest);
				System.out.println("[sync-resource] db/" + relative
						+ (copied ? " 복사 완료" : " 변경 없음, 건너뜀"));
			}
		}
	}

	// --- 공통 -------------------------------------------------------------------

	/** 크기·수정시각이 같으면 건너뛴다. 복사했으면 true. */
	private static boolean copyIfChanged(Path source, Path dest) throws IOException {
		if (Files.exists(dest)
				&& Files.size(source) == Files.size(dest)
				&& Files.getLastModifiedTime(source).equals(Files.getLastModifiedTime(dest))) {
			return false;
		}
		// COPY_ATTRIBUTES 로 수정시각까지 그대로 옮겨야 다음 실행에서 "변경 없음" 판정이 된다.
		Files.copy(source, dest, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.COPY_ATTRIBUTES);
		return true;
	}
}
