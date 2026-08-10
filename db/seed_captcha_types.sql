-- 캡차 타입 시드 데이터 (출처: 저장소 루트 data.json)
--
-- 적용: FastAPI 기동 시 init_db() 가 schema.sql 다음에 자동 실행한다
--       (경로는 DB_SEED_PATH, 기본 ./db/seed_captcha_types.sql).
--       수동 적용은 sqlite3 db/captchaSolver.sqlite3 < db/seed_captcha_types.sql
--
-- INSERT OR IGNORE 라 반복 실행해도 안전하다. 기존 행의 값을 고치지는 않으므로,
-- 값을 바꾸려면 해당 행을 지우고 다시 넣거나 UPDATE 를 쓴다.
--
-- data.json 의 `default`, `dev` 는 제외했다. 두 타입은 engine.get_captcha_type_list()
-- 에서 제거됐고 service_captchas 시드에도 없다.

PRAGMA foreign_keys = ON;

BEGIN;

-- ---------------------------------------------------------------------------
-- captcha_types: 표시용 이름/설명
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO captcha_types(captcha_id, name, description) VALUES
	('supreme_court', '대법원',  '대법원 캡챠'),
	-- data.json 은 '정부24' / '정부24 캡챠', engine.py 는 '정부 24' / '대한민국 정부 24 캡챠'.
	-- 여기서는 data.json 을 따랐다.
	('gov24',         '정부24',  '정부24 캡챠'),
	('wetax',         'WETAX',   'WETAX 캡챠'),
	('kshop',         'kshop',   'KT Shopping 캡챠');

-- ---------------------------------------------------------------------------
-- train_data_configs: 캡차별 학습/전처리 설정
--
-- 명시하지 않은 값은 data.json 의 train_data_defaults 와 동일하다.
--   backend='pytorch', rev=0, train_data_base_dir='./captcha_data',
--   image_width=200, image_height=50, label_length=6,
--   characters='0123456789', threshold=255
--
-- characters 는 data.json 의 train_data_defaults.characters 값이다. 네 타입 모두
-- 숫자 라벨이라 engine 이 파일명에서 감지하는 문자 집합과도 일치한다.
-- UNIQUE (captcha_id, backend, rev) 로 중복 삽입이 막힌다.
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO train_data_configs(
	captcha_id, backend, rev, train_data_base_dir,
	image_width, image_height, label_length, characters, threshold
) VALUES
	('supreme_court', 'pytorch', 0, './captcha_data', 120, 40, 6, '0123456789', 255),
	('gov24',         'pytorch', 1, './captcha_data', 200, 50, 6, '0123456789',  60),
	('wetax',         'pytorch', 0, './captcha_data', 200, 60, 6, '0123456789', 255),
	('kshop',         'pytorch', 0, './captcha_data', 263, 54, 6, '0123456789', 255);

-- ---------------------------------------------------------------------------
-- train_data_characters: 문자 단위로 쪼갠 표현 (문자별 정렬 순서가 필요할 때만 사용)
--
-- train_data_configs.characters 로 같은 정보를 문자열 하나에 담으므로 보통은
-- 시드할 필요가 없다. 문자 단위 행이 필요하면 아래 형태로 채운다:
--
--   INSERT OR IGNORE INTO train_data_characters(train_data_config_id, character, sort_order)
--   SELECT c.id, substr(c.characters, n.i, 1), n.i - 1
--   FROM train_data_configs c
--   JOIN (SELECT 1 AS i UNION ALL SELECT 2 UNION ALL SELECT 3 /* ... */) n
--     ON n.i <= length(c.characters)
--   WHERE c.captcha_id = 'supreme_court' AND c.backend = 'pytorch' AND c.rev = 0;
--
-- 이름 붙은 문자 집합 상수(DIGITS, ALPHA_NUMERIC 등)는 schema.sql 이 character_sets
-- 테이블에 함께 시드한다.
-- ---------------------------------------------------------------------------

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (5, 'seed_captcha_types');

COMMIT;
