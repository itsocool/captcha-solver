cmake_minimum_required(VERSION 3.24)
project(captcha-solver-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
	# /Gy /Gw = 함수/데이터를 각각 COMDAT 으로. 링커의 /OPT:REF,ICF 가 안 쓰는 걸 걷어낸다.
	add_compile_options(/utf-8 /Gy /Gw)  # /utf-8 은 소스의 한글 주석/문자열
endif()

set(ORT_VERSION "1.20.1" CACHE STRING "ONNX Runtime 버전")
set(CAPTCHA_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../captcha_data"
	CACHE PATH "모델을 가져올 captcha_data 디렉터리")
option(CAPTCHA_COPY_MODELS "빌드 산출물 옆에 captcha_data 의 최신 리비전 모델을 복사" ON)
set(CAPTCHA_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/config.json"
	CACHE FILEPATH "패키징 설정 JSON. \"captchas\" 배열이 설치할 유형을 정한다. 파일이 없으면 supreme_court 만")
option(CAPTCHA_SINGLE_EXE "onnxruntime.dll 을 압축해 실행 파일에 내장 (Windows 전용)" ON)

if(NOT WIN32)
	set(CAPTCHA_SINGLE_EXE OFF)
endif()
if(CAPTCHA_SINGLE_EXE)
	# CRT 까지 정적 링크. VC++ 재배포 패키지 없이 exe 하나로 돈다.
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ---------------------------------------------------------------- 의존성
# 둘 다 configure 시점에 자동으로 받아온다. vcpkg 불필요.
include(FetchContent)

if(WIN32)
	set(_ort_pkg "onnxruntime-win-x64-${ORT_VERSION}.zip")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
		set(_ort_pkg "onnxruntime-linux-aarch64-${ORT_VERSION}.tgz")
	else()
		set(_ort_pkg "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
	endif()
else()
	message(FATAL_ERROR "지원하지 않는 플랫폼입니다: ${CMAKE_SYSTEM_NAME} (Windows x64 / Linux x64·aarch64)")
endif()

FetchContent_Declare(ort
	URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ort_pkg}
	DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# ponytail: stb는 릴리스 태그가 없어 master를 쓴다. 재현성이 필요해지면 GIT_TAG를 커밋 해시로 고정.
FetchContent_Declare(stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG master
	GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(ort stb)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ort_SOURCE_DIR}/include")
if(WIN32)
	set_target_properties(onnxruntime PROPERTIES
		IMPORTED_LOCATION "${ort_SOURCE_DIR}/lib/onnxruntime.dll"
		IMPORTED_IMPLIB "${ort_SOURCE_DIR}/lib/onnxruntime.lib")
else()
	set_target_properties(onnxruntime PROPERTIES
		IMPORTED_LOCATION "${ort_SOURCE_DIR}/lib/libonnxruntime.so")
endif()

# ---------------------------------------------------------------- 타깃
# 전처리/디코딩은 실행 파일과 테스트가 같이 쓴다.
add_library(captcha_core STATIC src/preprocess.cpp src/decode.cpp)
target_include_directories(captcha_core PUBLIC src)
target_include_directories(captcha_core SYSTEM PUBLIC "${stb_SOURCE_DIR}")

add_executable(captcha src/main.cpp)
target_link_libraries(captcha PRIVATE captcha_core)
set_property(TARGET captcha captcha_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

if(CAPTCHA_SINGLE_EXE)
	# import lib 를 링크하지 않는다. 헤더만 쓰고 API 는 실행 시 GetProcAddress 로 받는다.
	target_include_directories(captcha SYSTEM PRIVATE "${ort_SOURCE_DIR}/include")
	target_compile_definitions(captcha PRIVATE CAPTCHA_SINGLE_EXE)
	target_link_libraries(captcha PRIVATE Cabinet.lib)  # 압축 해제(Compression API)

	# DLL 을 LZMS 로 압축해 RCDATA 리소스로 넣는다. 11.5MB -> 4MB 안팎.
	set(_ort_blob "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.lzms")
	add_executable(packblob tools/pack.cpp)
	target_link_libraries(packblob PRIVATE Cabinet.lib)
	add_custom_command(OUTPUT "${_ort_blob}"
		COMMAND packblob "${ort_SOURCE_DIR}/lib/onnxruntime.dll" "${_ort_blob}"
		DEPENDS packblob "${ort_SOURCE_DIR}/lib/onnxruntime.dll"
		COMMENT "onnxruntime.dll 압축")
	add_custom_target(ort_blob DEPENDS "${_ort_blob}")
	add_dependencies(captcha ort_blob)

	string(REPLACE "/" "\\\\" CAPTCHA_ORT_BLOB "${_ort_blob}")
	configure_file(src/ort_blob.rc.in "${CMAKE_CURRENT_BINARY_DIR}/ort_blob.rc" @ONLY)
	set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/ort_blob.rc" PROPERTIES OBJECT_DEPENDS "${_ort_blob}")
	target_sources(captcha PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/ort_blob.rc")
else()
	target_link_libraries(captcha PRIVATE onnxruntime)
endif()

if(NOT WIN32)
	find_package(Threads REQUIRED)
	target_link_libraries(captcha PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
	# 런타임 .so 를 실행 파일 옆에서 찾게 해 배포를 통째로 옮길 수 있게 한다.
	set_target_properties(captcha PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()

# ONNX Runtime 런타임 라이브러리를 실행 파일 옆에 둔다.
# ponytail: copy_if_different 는 심볼릭 링크를 따라가므로 리눅스에서는 .so 실체가 2벌 복사된다
#           (soname 이 어느 쪽이든 맞도록). 용량이 문제되면 create_symlink 로 바꿀 것.
if(NOT CAPTCHA_SINGLE_EXE)
	if(WIN32)
		file(GLOB _ort_runtime "${ort_SOURCE_DIR}/lib/*.dll")
	else()
		file(GLOB _ort_runtime "${ort_SOURCE_DIR}/lib/libonnxruntime.so*")
	endif()
	foreach(_lib ${_ort_runtime})
		add_custom_command(TARGET captcha POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_lib}" "$<TARGET_FILE_DIR:captcha>")
	endforeach()
endif()

# captcha_data/<id>/<rev>/model/ 의 model.ort + model.meta.json 을 <id>.ort / <id>.meta.json 으로
# 복사한다. 변환은 학습 쪽 finalize_artifacts() 가 이미 끝냈으므로 여기서는 이름만 바꾼다.
if(CAPTCHA_COPY_MODELS)
	# 담을 캡차 유형은 외부 config.json 의 "captchas" 배열이 정한다. 파일이 없으면 supreme_court 만.
	set(_wanted "supreme_court")
	if(EXISTS "${CAPTCHA_CONFIG_FILE}")
		set(_wanted "")
		set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CAPTCHA_CONFIG_FILE}")
		file(READ "${CAPTCHA_CONFIG_FILE}" _json)
		string(JSON _n ERROR_VARIABLE _err LENGTH "${_json}" captchas)
		if(NOT _err STREQUAL "NOTFOUND")
			message(FATAL_ERROR "${CAPTCHA_CONFIG_FILE} 의 \"captchas\" 배열을 읽지 못했습니다: ${_err}")
		endif()
		if(_n EQUAL 0)
			message(FATAL_ERROR "${CAPTCHA_CONFIG_FILE} 의 \"captchas\" 가 비어 있습니다.")
		endif()
		math(EXPR _last "${_n} - 1")
		foreach(_i RANGE ${_last})
			string(JSON _id GET "${_json}" captchas ${_i})
			list(APPEND _wanted "${_id}")
		endforeach()
		message(STATUS "캡차 유형 목록(${CAPTCHA_CONFIG_FILE}): ${_wanted}")
	else()
		message(STATUS "${CAPTCHA_CONFIG_FILE} 없음 — 기본값 ${_wanted} 만 담습니다.")
	endif()

	file(GLOB _id_dirs CONFIGURE_DEPENDS "${CAPTCHA_DATA_DIR}/*")
	set(_copied "")
	set(_copied_ids "")
	foreach(_id_dir ${_id_dirs})
		if(NOT IS_DIRECTORY "${_id_dir}")
			continue()
		endif()
		get_filename_component(_id "${_id_dir}" NAME)
		if(_wanted AND NOT "${_id}" IN_LIST _wanted)
			continue()
		endif()

		# 리비전은 숫자 디렉터리다. NATURAL 정렬의 마지막이 최신.
		file(GLOB _revs CONFIGURE_DEPENDS "${_id_dir}/*")
		set(_rev_names "")
		foreach(_rev ${_revs})
			get_filename_component(_rev_name "${_rev}" NAME)
			if(IS_DIRECTORY "${_rev}" AND _rev_name MATCHES "^[0-9]+$")
				list(APPEND _rev_names "${_rev_name}")
			endif()
		endforeach()
		# 원소가 "0" 하나뿐이면 if(NOT _rev_names) 가 참이 된다(CMake 의 거짓 상수). 길이로 본다.
		list(LENGTH _rev_names _rev_count)
		if(_rev_count EQUAL 0)
			continue()
		endif()
		list(SORT _rev_names COMPARE NATURAL)
		list(GET _rev_names -1 _rev)

		set(_model "${_id_dir}/${_rev}/model/model.ort")
		set(_meta "${_id_dir}/${_rev}/model/model.meta.json")
		if(NOT EXISTS "${_model}" OR NOT EXISTS "${_meta}")
			message(STATUS "모델 없음, 복사 건너뜀: ${_id} rev=${_rev}")
			continue()
		endif()

		add_custom_command(TARGET captcha POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_model}" "$<TARGET_FILE_DIR:captcha>/${_id}.ort"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_meta}" "$<TARGET_FILE_DIR:captcha>/${_id}.meta.json")
		install(FILES "${_model}" DESTINATION bin RENAME "${_id}.ort")
		install(FILES "${_meta}" DESTINATION bin RENAME "${_id}.meta.json")
		list(APPEND _copied "${_id}(rev=${_rev})")
		list(APPEND _copied_ids "${_id}")
	endforeach()
	message(STATUS "복사할 모델: ${_copied}")

	# 목록에 적었는데 빠진 유형은 설치본에서 조용히 사라지므로 configure 를 세운다.
	foreach(_id ${_wanted})
		if(NOT "${_id}" IN_LIST _copied_ids)
			message(FATAL_ERROR "설치 목록의 '${_id}' 를 ${CAPTCHA_DATA_DIR} 에서 찾지 못했습니다 "
				"(디렉터리·리비전·model.ort/model.meta.json 확인).")
		endif()
	endforeach()
endif()

# ---------------------------------------------------------------- 설치 / 패키징
# 실행 파일과 모델을 bin/ 에 나란히 둔다. main.cpp 가 모델을 <실행 파일 폴더> 에서 찾으므로
# 이 배치가 그대로 실행 조건이다. (모델 install 규칙은 위 복사 루프에 함께 있다)
install(TARGETS captcha RUNTIME DESTINATION bin)
if(NOT CAPTCHA_SINGLE_EXE)
	install(FILES ${_ort_runtime} DESTINATION bin)
endif()

set(CPACK_PACKAGE_NAME "captcha-solver")
set(CPACK_PACKAGE_VENDOR "HiWorks")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "캡차 인식 콘솔 도구")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "captcha-solver")
if(WIN32)
	set(CPACK_GENERATOR "NSIS")
	set(CPACK_NSIS_PACKAGE_NAME "captcha-solver")
	set(CPACK_NSIS_DISPLAY_NAME "captcha-solver")
	set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")  # x64 빌드. 기본값은 32비트 Program Files 다
	set(CPACK_NSIS_MODIFY_PATH ON)                  # 콘솔 도구라 bin 을 PATH 에 넣을지 묻는다
	set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
	# ponytail: 압축기는 기본값(zlib) 그대로. .ort 는 이미 조밀해서 lzma 로 바꿔도 크기가 같았다.
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../hi.works.ico")
		file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../hi.works.ico" _icon)
		set(CPACK_NSIS_MUI_ICON "${_icon}")
		set(CPACK_NSIS_MUI_UNIICON "${_icon}")
	endif()
endif()
include(CPack)
