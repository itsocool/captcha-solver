-- 캡차 타입 시드 데이터 (출처: 저장소 루트 data.json)
--
-- 적용: sqlite3 db/captchaSolver.sqlite3 < db/seed_captcha_types.sql
--       (schema.sql 이 먼저 적용되어 있어야 한다)
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
--   image_width=200, image_height=50, label_length=6, threshold=255
-- UNIQUE (captcha_id, backend, rev) 로 중복 삽입이 막힌다.
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO train_data_configs(
	captcha_id, backend, rev, train_data_base_dir,
	image_width, image_height, label_length, threshold
) VALUES
	('supreme_court', 'pytorch', 0, './captcha_data', 120, 40, 6, 255),
	('gov24',         'pytorch', 1, './captcha_data', 200, 50, 6,  60),
	('wetax',         'pytorch', 0, './captcha_data', 200, 60, 6, 255),
	('kshop',         'pytorch', 0, './captcha_data', 263, 54, 6, 255);

-- ---------------------------------------------------------------------------
-- train_data_characters: 캡차별 문자 집합
--
-- 위 네 타입은 문자 집합을 명시하지 않는다. TrainData 가 images/train 의 파일명에서
-- 문자 집합과 라벨 길이를 자동 감지하므로 시드할 행이 없다.
-- 고정 문자 집합을 넣어야 한다면 아래 형태로 추가한다:
--
--   INSERT OR IGNORE INTO train_data_characters(train_data_config_id, character, sort_order)
--   SELECT c.id, s.value, s.key
--   FROM train_data_configs c
--   JOIN json_each(json_array('0','1','2')) s
--   WHERE c.captcha_id = 'example' AND c.backend = 'pytorch' AND c.rev = 0;
-- ---------------------------------------------------------------------------

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (3, 'seed_captcha_types');

COMMIT;
