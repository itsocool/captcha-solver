-- 캡차 타입 시드 데이터
--
-- 원래 저장소 루트 data.json 에 있던 값이다. data.json 은 제거됐고 이 파일이
-- 유일한 출처다. 런타임 레지스트리는 engine.get_captcha_type_list() 쪽이다.
--
-- 적용: FastAPI 기동 시 init_db() 가 schema.sql 다음에 자동 실행한다
--       (경로는 DB_SEED_PATH, 기본 ./db/seed_captcha_types.sql).
--       수동 적용은 sqlite3 db/captchaSolver.sqlite3 < db/seed_captcha_types.sql
--
-- INSERT OR IGNORE 라 반복 실행해도 안전하다. 기존 행의 값을 고치지는 않으므로,
-- 값을 바꾸려면 해당 행을 지우고 다시 넣거나 UPDATE 를 쓴다.
--
-- 옛 data.json 의 `default`, `dev` 는 제외했다. 두 타입은 engine.get_captcha_type_list()
-- 에서 제거됐고 service_captchas 시드에도 없다.

PRAGMA foreign_keys = ON;

BEGIN;

-- ---------------------------------------------------------------------------
-- 마이그레이션 8: 리비전은 1부터 시작한다 / kshop 캡차 제거
--
-- 예전에는 기본 리비전이 0 이었다. 디렉터리(captcha_data/<id>/0 → /1)와 함께 DB 의
-- rev 0 행을 1 로 옮긴다. UPDATE OR IGNORE 라 이미 rev 1 행이 있는 캡차는 건너뛰고,
-- 반복 실행해도 안전하다 (두 번째부터는 rev 0 행이 없어 no-op).
-- kshop 은 레지스트리(engine.get_captcha_type_list())에서 제거됐다. captcha_types 를
-- 지우면 train_data_configs / train_run_params 는 FK ON DELETE CASCADE 로 함께 지워지고,
-- service_captchas 는 FK 가 없어 직접 지운다.
-- ---------------------------------------------------------------------------
DELETE FROM captcha_types    WHERE captcha_id = 'kshop';
DELETE FROM service_captchas WHERE captcha_id = 'kshop';
UPDATE OR IGNORE train_data_configs SET rev = 1 WHERE rev = 0;
UPDATE OR IGNORE train_run_params   SET rev = 1 WHERE rev = 0;
INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (8, 'rev_starts_at_1_drop_kshop');

-- ---------------------------------------------------------------------------
-- captcha_types: 표시용 이름/설명
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO captcha_types(captcha_id, name, description) VALUES
	('supreme_court', '대법원',  '대법원 캡챠'),
	-- 옛 data.json 은 '정부24' / '정부24 캡챠', engine.py 는 '정부 24' / '대한민국 정부 24 캡챠'.
	-- 여기서는 data.json 값을 따랐다.
	('gov24',         '정부24',  '정부24 캡챠'),
	('wetax',         'WETAX',   'WETAX 캡챠'),
	('iptime',        'ipTIME',  'ipTIME 공유기 캡챠');

-- ---------------------------------------------------------------------------
-- train_data_configs: 캡차별 학습/전처리 설정
--
-- 명시하지 않은 값은 옛 data.json 의 train_data_defaults 와 동일하다.
--   backend='pytorch', rev=1, train_data_base_dir='./captcha_data',
--   image_width=200, image_height=50, label_length=6,
--   characters='0123456789', threshold=255
--
-- characters 는 옛 data.json 의 train_data_defaults.characters 값이다. iptime 을 뺀
-- 타입은 숫자 라벨이라 engine 이 파일명에서 감지하는 문자 집합과도 일치한다.
-- UNIQUE (captcha_id, backend, rev) 로 중복 삽입이 막힌다.
-- ---------------------------------------------------------------------------
-- image_width/image_height 는 크롭 **전** 원본 크기다 (crop 좌표계). preprocess/crop 은
-- engine.get_captcha_type_list() 의 값과 일치해야 한다. crop 은 PIL [left,top,right,bottom].
INSERT OR IGNORE INTO train_data_configs(
	captcha_id, backend, rev, train_data_base_dir,
	image_width, image_height, label_length, characters, threshold, preprocess, crop
) VALUES
	('supreme_court', 'pytorch', 1, './captcha_data', 120, 40, 6, '0123456789', 255, 'supreme_court', NULL),
	('gov24',         'pytorch', 1, './captcha_data', 200, 50, 6, '0123456789',  60, 'default', NULL),
	('wetax',         'pytorch', 1, './captcha_data', 200, 60, 6, '0123456789', 255, 'default', NULL),
	('iptime',        'pytorch', 1, './captcha_data', 200, 70, 5, 'abcdefghijklmnopqrstuvwxyz', 255, 'iptime', '[27, 10, 195, 70]');

-- 이미 시드된 행은 INSERT OR IGNORE 가 건드리지 않으므로 preprocess/crop 을 UPDATE 로 보정한다.
-- UPDATE 는 멱등이라 반복 실행해도 안전하다 (마이그레이션 6: preprocess/crop 컬럼 추가분).
UPDATE train_data_configs SET preprocess = 'supreme_court'
	WHERE captcha_id = 'supreme_court' AND backend = 'pytorch' AND rev = 1;

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
--   WHERE c.captcha_id = 'supreme_court' AND c.backend = 'pytorch' AND c.rev = 1;
--
-- 이름 붙은 문자 집합 상수(DIGITS, ALPHA_NUMERIC 등)는 schema.sql 이 character_sets
-- 테이블에 함께 시드한다.
-- ---------------------------------------------------------------------------

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (5, 'seed_captcha_types');

COMMIT;
