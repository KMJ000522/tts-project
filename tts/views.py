# tts/views.py

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline
import soundfile as sf
import torch
import json
import io
import numpy as np

try:
    print("TTS 모델을 로딩합니다... (시간이 매우 오래 걸릴 수 있습니다)")
    pipe = pipeline("text-to-speech", 
                    model="suno/bark-small")
    print("✅ TTS 모델 로딩 완료.")
    model_loaded = True
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    pipe = None
    model_loaded = False

@csrf_exempt
def generate_speech(request):
    if not model_loaded:
        return JsonResponse({'error': '서버의 TTS 모델이 로드되지 않았습니다.'}, status=500)

    if request.method == 'POST':
        data = json.loads(request.body)
        text_to_speak = data.get('text', '')

        if not text_to_speak:
            return JsonResponse({'error': '음성으로 변환할 텍스트가 없습니다.'}, status=400)

        try:
            print(f"'{text_to_speak}' 텍스트로 음성 생성 시작...")
            output = pipe(text_to_speak)
            
            speech_data = output["audio"]
            sampling_rate = output["sampling_rate"]

            # <<< 1. 이 줄이 핵심적인 변경점입니다.
            # 모델이 2차원 배열([[...]])로 데이터를 주므로, 실제 음성 데이터인 첫 번째 배열을 꺼냅니다.
            if len(speech_data.shape) > 1:
                speech_data = speech_data[0]

            # <<< 2. 이 줄은 데이터의 재질(타입)을 확실하게 보장해줍니다.
            speech_data = np.array(speech_data, dtype=np.float32)

            buffer = io.BytesIO()
            sf.write(buffer, speech_data, sampling_rate, format='WAV')
            buffer.seek(0)

            print("✅ 음성 생성 완료. 파일을 전송합니다.")
            
            response = HttpResponse(buffer, content_type='audio/wav')
            response['Content-Disposition'] = 'attachment; filename="output.wav"'
            
            return response

        except Exception as e:
            print(f"❌ 음성 생성 중 오류 발생: {e}")
            return JsonResponse({'error': f'음성 생성 중 오류 발생: {e}'}, status=500)

    return JsonResponse({'error': 'POST 요청만 지원합니다.'}, status=405)