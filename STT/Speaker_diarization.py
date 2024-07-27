from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.database.util import load_rttm
import torch
import torchaudio

def create_rttm(waveform, sample_rate, dest_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token="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")

    def device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


    pipeline.to(device())

    # waveform, sample_rate = torchaudio.load(aud_path)

    with ProgressHook() as hook:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
        

    with open(dest_path, "w") as rttm:
        diarization.write_rttm(rttm)
    
    return load_rttm(dest_path)