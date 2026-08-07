cmake_minimum_required(VERSION 3.24)
project(captcha-solver-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
	add_compile_options(/utf-8)  # 소스의 한글 주석/문자열
endif()

set(ORT_VERSION "1.20.1" CACHE STRING "ONNX Runtime 버전")
option(CAPTCHA_COPY_MODELS "빌드 산출물 옆에 ConsoleApp.v4 모델 + cli 메타를 복사" ON)

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
target_link_libraries(captcha PRIVATE captcha_core onnxruntime)

if(NOT WIN32)
	find_package(Threads REQUIRED)
	target_link_libraries(captcha PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
	# 런타임 .so 를 실행 파일 옆에서 찾게 해 배포를 통째로 옮길 수 있게 한다.
	set_target_properties(captcha PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()

# ONNX Runtime 런타임 라이브러리를 실행 파일 옆에 둔다.
# ponytail: copy_if_different 는 심볼릭 링크를 따라가므로 리눅스에서는 .so 실체가 2벌 복사된다
#           (soname 이 어느 쪽이든 맞도록). 용량이 문제되면 create_symlink 로 바꿀 것.
if(WIN32)
	file(GLOB _ort_runtime "${ort_SOURCE_DIR}/lib/*.dll")
else()
	file(GLOB _ort_runtime "${ort_SOURCE_DIR}/lib/libonnxruntime.so*")
endif()
foreach(_lib ${_ort_runtime})
	add_custom_command(TARGET captcha POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_lib}" "$<TARGET_FILE_DIR:captcha>")
endforeach()

# apps/cli/models 의 메타데이터를 기준으로 모델을 훑는다.
# 모델 본체는 ConsoleApp.v4 의 <id>.model 을 우선 쓰고, 없으면 cli 의 <id>.onnx 를 쓴다.
if(CAPTCHA_COPY_MODELS)
	set(_cli_models "${CMAKE_CURRENT_SOURCE_DIR}/../cli/models")
	file(GLOB _metas CONFIGURE_DEPENDS "${_cli_models}/*.meta.json")
	set(_copied "")
	foreach(_meta ${_metas})
		get_filename_component(_name "${_meta}" NAME)
		string(REGEX REPLACE "\\.meta\\.json$" "" _id "${_name}")

		set(_model "${CMAKE_CURRENT_SOURCE_DIR}/../ConsoleApp.v4/${_id}.model")
		if(NOT EXISTS "${_model}")
			set(_model "${_cli_models}/${_id}.onnx")
		endif()
		if(NOT EXISTS "${_model}")
			message(STATUS "모델 없음, 복사 건너뜀: ${_id}")
			continue()
		endif()

		add_custom_command(TARGET captcha POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_model}" "$<TARGET_FILE_DIR:captcha>/${_id}.model"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_meta}" "$<TARGET_FILE_DIR:captcha>/${_id}.meta.json")
		list(APPEND _copied ${_id})
	endforeach()
	message(STATUS "복사할 모델: ${_copied}")
endif()

# ---------------------------------------------------------------- 테스트
include(CTest)
if(BUILD_TESTING)
	add_executable(test_decode tests/test_decode.cpp)
	target_link_libraries(test_decode PRIVATE captcha_core)
	add_test(NAME decode COMMAND test_decode)

	# apps/cli/samples/<id>/<정답>.png 로 end-to-end 확인.
	add_test(NAME samples COMMAND ${CMAKE_COMMAND}
		-DEXE=$<TARGET_FILE:captcha>
		-DSAMPLES=${CMAKE_CURRENT_SOURCE_DIR}/../cli/samples
		-P ${CMAKE_CURRENT_SOURCE_DIR}/tests/samples.cmake)
endif()
