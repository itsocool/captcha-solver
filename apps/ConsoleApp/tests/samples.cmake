# apps/cli/samples/<id>/<정답>.png 를 전부 인식해 파일명과 대조한다.
# ctest -R samples 로 실행. EXE / SAMPLES 는 CMakeLists가 넘긴다.
#
# dev 는 ONNX(200×50)와 메타(250×50) 크기가 어긋나 추론 자체가 안 돼서 뺐다(apps/cli/README.md).
# 나머지 5종은 샘플 10장씩 전부 일치한다.
if(NOT DEFINED IDS)
	set(IDS gov24 supreme_court kshop wetax default)
endif()

set(total 0)
set(fail 0)

foreach(id ${IDS})
	get_filename_component(_exe_dir "${EXE}" DIRECTORY)
	if(NOT EXISTS "${_exe_dir}/${id}.model")
		message(STATUS "${id}: 모델이 없어 건너뜁니다 (-DCAPTCHA_COPY_MODELS=ON 으로 빌드하세요)")
		continue()
	endif()

	file(GLOB images "${SAMPLES}/${id}/*.png" "${SAMPLES}/${id}/*.jpg")
	if(NOT images)
		message(STATUS "${id}: 샘플이 없어 건너뜁니다 (${SAMPLES}/${id})")
		continue()
	endif()

	foreach(img ${images})
		get_filename_component(expected "${img}" NAME_WE)
		execute_process(
			COMMAND "${EXE}" -c=${id} -i=${img}
			OUTPUT_VARIABLE got
			ERROR_VARIABLE err
			RESULT_VARIABLE rc)
		math(EXPR total "${total} + 1")
		if(NOT rc EQUAL 0)
			message("FAIL ${id}/${expected}: exit ${rc} ${err}")
			math(EXPR fail "${fail} + 1")
		elseif(NOT got STREQUAL expected)
			message("MISMATCH ${id}/${expected}: got '${got}'")
			math(EXPR fail "${fail} + 1")
		endif()
	endforeach()
endforeach()

if(total EQUAL 0)
	message(FATAL_ERROR "검사한 샘플이 없습니다")
endif()
math(EXPR ok "${total} - ${fail}")
message(STATUS "samples: ${ok}/${total} 일치")
if(fail GREATER 0)
	message(FATAL_ERROR "${fail}건 불일치")
endif()
