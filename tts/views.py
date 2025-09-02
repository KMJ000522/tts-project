import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile # pyright: ignore[reportMissingImports]
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import time

# --- 모델 로딩 (서버 시작 시 1회만 실행) ---
try:
    print("⏳ TTS 모델 로딩을 시작합니다...")
    start_time = time.time()
    
    model = VitsModel.from_pretrained("facebook/mms-tts-kor")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    
    end_time = time.time()
    print(f"✅ TTS 모델 로딩 완료. 소요 시간: {end_time - start_time:.2f}초")

except Exception as e:
    print(f"🔥 TTS 모델 로딩 중 오류 발생: {e}")
    model = None
    tokenizer = None
# -----------------------------------------

@csrf_exempt
def health_check(request):
    """
    Render.com의 Health Check를 위한 뷰.
    모델이 성공적으로 로드되었는지 여부에 따라 상태를 반환합니다.
    """
    if model and tokenizer:
        return JsonResponse({"status": "ok", "message": "Model is loaded."})
    else:
        # 503 Service Unavailable
        return JsonResponse({"status": "error", "message": "Model is not loaded."}, status=503)

@csrf_exempt
def tts_view(request):
    """
    메인 TTS 요청을 처리하는 뷰.
    POST 요청으로 텍스트를 받아 음성 WAV 파일을 반환합니다.
    """
    if request.method == 'POST':
        # 모델 로딩 실패 시 에러 반환
        if not model or not tokenizer:
            return JsonResponse({"error": "TTS 모델이 준비되지 않았습니다."}, status=500)

        try:
            # 요청 본문(body)에서 JSON 데이터 파싱
            data = json.loads(request.body)
            text = data.get('text')

            if not text:
                return JsonResponse({"error": "텍스트를 입력해주세요."}, status=400)

            print(f"요청 수신: {text}")

            # TTS 변환 시작
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform

            # NumPy 배열로 변환 및 정규화 (16-bit PCM 형식으로)
            speech_np = output.squeeze().numpy()
            speech_np = np.int16(speech_np / np.max(np.abs(speech_np)) * 32767)
            
            # WAV 파일로 응답 생성
            sampling_rate = model.config.sampling_rate
            
            # 메모리에서 WAV 데이터 생성
            from io import BytesIO
            buffer = BytesIO()
            scipy.io.wavfile.write(buffer, rate=sampling_rate, data=speech_np)
            buffer.seek(0) # 버퍼의 시작으로 포인터 이동

            # HttpResponse로 WAV 파일 반환
            response = HttpResponse(buffer, content_type='audio/wav')
            response['Content-Disposition'] = 'attachment; filename="output.wav"'
            
            print("✅ 음성 변환 및 전송 완료")
            return response

        except json.JSONDecodeError:
            return JsonResponse({"error": "잘못된 JSON 형식입니다."}, status=400)
        except Exception as e:
            print(f"🔥 TTS 처리 중 오류 발생: {e}")
            return JsonResponse({"error": f"서버 내부 오류: {str(e)}"}, status=500)

    # POST 요청이 아닐 경우
    return JsonResponse({"error": "POST 요청만 지원합니다."}, status=405)