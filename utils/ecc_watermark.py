import bchlib
import numpy as np
import torch
from scipy.stats import norm

class ECCWatermarker:
    def __init__(self, wm_text="test", bch_bits=8, repetition=3, latent_shape=(4, 64, 64)):
        """
        Implementation of Robust Watermarking using ECC and Distribution-Aware Sampling.
        """
        self.wm_text = wm_text
        self.repetition = repetition
        self.latent_shape = latent_shape
        
        # 1. Initialize BCH
        # __init__(t, poly=None, m=None)
        # t: number of errors to correct (bch_bits)
        self.bch_t = bch_bits
        self.bch_m = 10 # 2^10 - 1 = 1023 bits max codeword
        
        try:
            # Note: The first argument is t, NOT poly.
            self.bch = bchlib.BCH(self.bch_t, m=self.bch_m)
        except Exception as e:
            print(f"BCH Init Error: {e}")
            # Fallback to m=13 if m=10 fails for some reason
            self.bch = bchlib.BCH(self.bch_t, m=13)
        
    def _text_to_bits(self, text):
        data = bytearray(text, 'utf-8')
        ecc = self.bch.encode(data)
        packet = data + ecc
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        return bits

    def _bits_to_text(self, bits):
        # Ensure bits length is a multiple of 8
        if len(bits) % 8 != 0:
            bits = np.concatenate([bits, np.zeros(8 - (len(bits) % 8), dtype=np.uint8)])
            
        packet = np.packbits(bits).tobytes()
        data_len = len(packet) - self.bch.ecc_bytes
        data = bytearray(packet[:data_len])
        ecc = bytearray(packet[data_len:])
        
        bitflips = self.bch.decode(data, ecc)
        if bitflips >= 0:
            self.bch.correct(data, ecc)
            try:
                decoded = data.decode('utf-8', errors='ignore')
                return decoded[:len(self.wm_text)], bitflips
            except:
                return None, -1
        return None, -1

    def encode(self):
        bits = self._text_to_bits(self.wm_text)
        repeated_bits = np.repeat(bits, self.repetition)
        return repeated_bits

    def map_to_latent(self, bits, seed=42):
        np.random.seed(seed)
        total_elements = np.prod(self.latent_shape)
        if len(bits) > total_elements:
            raise ValueError(f"Watermark bits ({len(bits)}) exceed latent capacity ({total_elements})")

        latent = np.random.normal(0, 1, self.latent_shape).astype(np.float32)
        flat_latent = latent.flatten()
        
        u = np.random.uniform(0, 1, len(bits))
        mapped_values = norm.ppf((u + bits) / 2.0)
        flat_latent[:len(bits)] = mapped_values
        
        return torch.from_numpy(flat_latent.reshape(self.latent_shape))

    def extract_from_latent(self, latent_tensor):
        if isinstance(latent_tensor, torch.Tensor):
            latent = latent_tensor.cpu().numpy()
        else:
            latent = latent_tensor
            
        flat_latent = latent.flatten()
        num_encoded_bits = len(self.encode())
        probs = norm.cdf(flat_latent[:num_encoded_bits])
        extracted_bits = (probs > 0.5).astype(np.uint8)
        
        reshaped = extracted_bits.reshape(-1, self.repetition)
        voted_bits = (np.mean(reshaped, axis=1) > 0.5).astype(np.uint8)
        
        text, bitflips = self._bits_to_text(voted_bits)
        return text, bitflips

if __name__ == "__main__":
    wm_msg = "ECC_ROBUST"
    wm = ECCWatermarker(wm_text=wm_msg, bch_bits=8, repetition=5)
    
    bits = wm.encode()
    print(f"Original Text: {wm_msg}")
    print(f"Code Length (bits): {len(bits)}")
    
    latent = wm.map_to_latent(bits)
    
    # Test with heavy noise
    for noise_level in [0.1, 0.3, 0.5]:
        noise = torch.randn_like(latent) * noise_level
        noisy_latent = latent + noise
        extracted_text, flips = wm.extract_from_latent(noisy_latent)
        print(f"Noise {noise_level} | Extracted: {extracted_text} | Flips: {flips}")
