import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile
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
    if model and tokenizer:
        return JsonResponse({"status": "ok", "message": "Model is loaded."})
    else:
        return JsonResponse({"status": "error", "message": "Model is not loaded."}, status=503)

@csrf_exempt
def tts_view(request):
    if request.method == 'POST':
        if not model or not tokenizer:
            return JsonResponse({"error": "TTS 모델이 준비되지 않았습니다."}, status=500)

        try:
            data = json.loads(request.body)
            text = data.get('text')

            # --- 🕵️‍♂️ 디버깅용 블랙박스 코드 ---
            print("--- BLACKBOX ---")
            print(f"Received Raw Body Type: {type(request.body)}")
            print(f"Received Text: {text}")
            print(f"Text Type: {type(text)}")
            if text:
                print(f"Text as Bytes (UTF-8): {text.encode('utf-8')}")
            print("--- END BLACKBOX ---")
            # ------------------------------------

            if not text or not text.strip():
                return JsonResponse({"error": "내용이 없는 텍스트는 변환할 수 없습니다."}, status=400)

            inputs = tokenizer(text, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].to(torch.long)
            
            with torch.no_grad():
                output = model(**inputs).waveform

            speech_np = output.squeeze().numpy()
            speech_np = np.int16(speech_np / np.max(np.abs(speech_np)) * 32767)
            
            sampling_rate = model.config.sampling_rate
            
            from io import BytesIO
            buffer = BytesIO()
            scipy.io.wavfile.write(buffer, rate=sampling_rate, data=speech_np)
            buffer.seek(0)

            response = HttpResponse(buffer, content_type='audio/wav')
            response['Content-Disposition'] = 'attachment; filename="output.wav"'
            
            print("✅ 음성 변환 및 전송 완료")
            return response

        except Exception as e:
            print(f"🔥 TTS 처리 중 오류 발생: {e}")
            return JsonResponse({"error": f"서버 내부 오류: {str(e)}"}, status=500)

    return JsonResponse({"error": "POST 요청만 지원합니다."}, status=405)