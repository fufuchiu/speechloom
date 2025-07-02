import numpy as np

from speechloom.codec import mulaw_decode, mulaw_encode
from speechloom.metrics import waveform_snr
from speechloom.tokens import pack_response, unpack_response

waveform = np.sin(np.arange(320) * 0.1) * 0.5
codes = mulaw_encode(waveform)
packed = pack_response('你好', codes)
text, restored_codes = unpack_response(packed)
restored = mulaw_decode(restored_codes)
print({'text': text, 'tokens': len(packed), 'snr_db': waveform_snr(waveform, restored)})
assert text == '你好'
