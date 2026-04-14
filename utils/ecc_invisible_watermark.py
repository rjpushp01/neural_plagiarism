import cv2
import bchlib
import numpy as np
from imwatermark import WatermarkEncoder, WatermarkDecoder


class ECCInvisibleWatermarker:
    """
    Combines spatial/frequency pixel watermarking (dwtDctSvd) with robust ECC.
    This provides Option B: Zero hallucination (modifies pixels directly) and
    high resistance against Shims via heavy repetition and BCH correction.
    """
    
    def __init__(self, wm_text="test", bch_bits=5, repetition=5, method='dwtDctSvd'):
        self.wm_text = wm_text
        self.bch_t = bch_bits
        self.bch_m = 10
        self.repetition = repetition
        self.method = method
        
        try:
            self.bch = bchlib.BCH(self.bch_t, m=self.bch_m)
        except Exception as e:
            self.bch = bchlib.BCH(self.bch_t, m=13)
            
        # We need to calculate what the expected byte array length will be.
        dummy_bits = self._text_to_bits(self.wm_text)
        self.original_bits_len = len(dummy_bits)
        repeated_bits = np.repeat(dummy_bits, self.repetition)
        
        # Pad to bytes
        if len(repeated_bits) % 8 != 0:
            padded_bits = np.concatenate([repeated_bits, np.zeros(8 - (len(repeated_bits) % 8), dtype=np.uint8)])
        else:
            padded_bits = repeated_bits
            
        self.payload_bytes_len = len(padded_bits) // 8
        
        # The underlying invisible watermarker works on bits directly but expects a length
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder('bytes', self.payload_bytes_len * 8)

    def _text_to_bits(self, text):
        data = bytearray(text, 'utf-8')
        ecc = self.bch.encode(data)
        packet = data + ecc
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        return bits

    def _bits_to_text(self, bits):
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

    def _create_payload_bytes(self):
        bits = self._text_to_bits(self.wm_text)
        repeated_bits = np.repeat(bits, self.repetition)
        
        if len(repeated_bits) % 8 != 0:
            padded_bits = np.concatenate([repeated_bits, np.zeros(8 - (len(repeated_bits) % 8), dtype=np.uint8)])
        else:
            padded_bits = repeated_bits
            
        return np.packbits(padded_bits).tobytes()

    def encode(self, img_path, output_path):
        """Embed the ECC-protected payload into the image."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
            
        payload_bytes = self._create_payload_bytes()
        
        self.encoder.set_watermark('bytes', payload_bytes)
        out = self.encoder.encode(img, self.method)
        cv2.imwrite(output_path, out)

    def decode(self, img_path):
        """Extract the ECC-protected payload and decode the text."""
        wm_img = cv2.imread(img_path)
        if wm_img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        # Ensure minimum size requested by imwatermark
        h, w = wm_img.shape[:2]
        if h <= 256 or w <= 256:
            wm_img = cv2.resize(wm_img, (512, 512), interpolation=cv2.INTER_CUBIC)
            
        wm_img = wm_img.astype(np.float32)
        
        try:
            # imwatermark returns a standard string if it can decode, or raw bytes under 'bytes' method
            extracted_bytes = self.decoder.decode(wm_img, self.method)
            # DWTDCTSVD via 'bytes' mode actually returns a byte string
            if isinstance(extracted_bytes, str):
                extracted_bytes = extracted_bytes.encode('utf-8')
        except Exception as e:
            print(f"Decoding failed structurally: {e}")
            return {'text': None, 'ber_final': 1.0, 'message_recovered': False}
            
        # Take the length we actually expect; pad if it's too short
        if len(extracted_bytes) < self.payload_bytes_len:
            extracted_bytes = extracted_bytes + b'\x00' * (self.payload_bytes_len - len(extracted_bytes))
        extracted_bytes = extracted_bytes[:self.payload_bytes_len]
        
        extracted_bits = np.unpackbits(np.frombuffer(extracted_bytes, dtype=np.uint8))
        
        # Remove any padding we added to make it byte-aligned
        target_len = self.original_bits_len * self.repetition
        extracted_bits = extracted_bits[:target_len]
        
        # Repetition Voting
        reshaped = extracted_bits.reshape(-1, self.repetition)
        vote_scores = np.mean(reshaped, axis=1)
        voted_bits = (vote_scores > 0.5).astype(np.uint8)
        
        # Original Bits calculation for BER
        original_bits = self._text_to_bits(self.wm_text)
        original_repeated = np.repeat(original_bits, self.repetition)
        ber_raw = float(np.mean(extracted_bits != original_repeated))
        ber_voted = float(np.mean(voted_bits != original_bits))
        
        text, bitflips = self._bits_to_text(voted_bits)
        message_recovered = (text == self.wm_text)
        
        ber_final = 0.0 if message_recovered else ber_voted
        
        return {
            'text': text,
            'bch_corrections': bitflips,
            'ber_raw': float(ber_raw),
            'ber_voted': float(ber_voted),
            'ber_final': float(ber_final),
            'message_recovered': message_recovered,
        }

if __name__ == "__main__":
    # Test script strictly to ensure dimensions and encoding works
    import os
    print("Testing ECCInvisibleWatermarker...")
    os.makedirs("/tmp/ecc_test", exist_ok=True)
    
    # Create simple dummy image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[100:400, 100:400] = 128
    cv2.imwrite("/tmp/ecc_test/dummy.png", img)
    
    wm = ECCInvisibleWatermarker(wm_text="test", bch_bits=5, repetition=5)
    wm.encode("/tmp/ecc_test/dummy.png", "/tmp/ecc_test/dummy_wm.png")
    
    res = wm.decode("/tmp/ecc_test/dummy_wm.png")
    print("Extraction successful:", res)
