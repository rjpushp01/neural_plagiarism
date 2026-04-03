import bchlib
import numpy as np
import torch
from scipy.stats import norm


class ECCWatermarker:
    """
    Robust Watermarking using ECC (BCH + Repetition Coding) and Distribution-Aware Sampling.

    Latent Channel Semantics (SD VAE):
        Channel 0: Coarse luminance/brightness — DO NOT modify (most visually sensitive)
        Channel 1: Color/chrominance — DO NOT modify
        Channel 2: Higher-frequency spatial detail — safe to modify (less perceptual impact)
        Channel 3: Fine texture/edges — safe to modify (less perceptual impact)

    By default, watermark bits are embedded only in channel 2 ('single' mode).
    When target_channels='dual', bits are split across channels 2 and 3, halving
    the per-channel modification density and improving PSNR.
    """

    def __init__(self, wm_text="test", bch_bits=8, repetition=3,
                 latent_shape=(4, 64, 64), ecc_seed=42, target_channels='single'):
        self.wm_text = wm_text
        self.repetition = repetition
        self.latent_shape = latent_shape
        self.ecc_seed = ecc_seed
        self.target_channels = target_channels  # 'single' (ch2 only) or 'dual' (ch2+ch3)

        # Initialize BCH
        self.bch_t = bch_bits
        self.bch_m = 10  # 2^10 - 1 = 1023 bits max codeword

        try:
            self.bch = bchlib.BCH(self.bch_t, m=self.bch_m)
        except Exception as e:
            print(f"BCH Init Error: {e}")
            self.bch = bchlib.BCH(self.bch_t, m=13)

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

    def encode(self):
        bits = self._text_to_bits(self.wm_text)
        repeated_bits = np.repeat(bits, self.repetition)
        return repeated_bits

    def get_capacity_info(self, latent_shape=None):
        """Return diagnostic info about watermark capacity for a given latent shape."""
        if latent_shape is None:
            latent_shape = self.latent_shape
        bits = self.encode()
        _, c, h, w = latent_shape if len(latent_shape) == 4 else (1, *latent_shape)
        channel_elements = h * w
        total_elements = c * h * w

        if self.target_channels == 'dual':
            available = 2 * channel_elements  # channels 2 + 3
        else:
            available = channel_elements  # channel 2 only

        return {
            'num_bits': len(bits),
            'available_elements': available,
            'utilization_pct': len(bits) / available * 100,
            'total_latent_pct': len(bits) / total_elements * 100,
            'fits': len(bits) <= available,
        }

    def _get_indices(self, latent_shape):
        """Compute scatter indices for embedding/extraction based on target_channels mode."""
        if len(latent_shape) == 4:
            _, c, h, w = latent_shape
        else:
            c, h, w = latent_shape

        channel_elements = h * w
        num_bits = len(self.encode())
        np.random.seed(self.ecc_seed)

        if self.target_channels == 'dual':
            # Split bits across channels 2 and 3 (both high-frequency, low perceptual impact)
            half = num_bits // 2
            remainder = num_bits - half  # handle odd number

            ch2_offset = 2 * channel_elements
            ch3_offset = 3 * channel_elements

            ch2_indices = np.random.choice(channel_elements, half, replace=False)
            ch3_indices = np.random.choice(channel_elements, remainder, replace=False)

            indices = np.concatenate([ch2_offset + ch2_indices, ch3_offset + ch3_indices])
        else:
            # Original: channel 2 only
            offset = 2 * channel_elements
            channel_indices = np.random.choice(channel_elements, num_bits, replace=False)
            indices = offset + channel_indices

        return indices

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

    def embed_into_latent(self, latent_tensor, seed=42, margin=0.75):
        """Embed watermark bits into an EXISTING latent tensor via sign-nudging.

        Instead of overwriting with random quantile-mapped values (which create
        visible blue-dot artifacts), this method preserves the original latent 
        magnitude and only adjusts the sign to encode each bit:
          - bit=1: ensure value > +margin  
          - bit=0: ensure value < -margin

        The margin controls the tradeoff between PSNR and extraction robustness.
        Larger margin = more robust but lower PSNR. Default 0.75 gives good balance.

        Extraction checks: norm.cdf(value) > 0.5, i.e. value > 0.
        """
        bits = self.encode()

        if isinstance(latent_tensor, torch.Tensor):
            latent = latent_tensor.clone().cpu().numpy()
        else:
            latent = np.copy(latent_tensor)

        flat_latent = latent.flatten()
        total_elements = flat_latent.shape[0]

        if len(bits) > total_elements:
            raise ValueError(f"Watermark bits ({len(bits)}) exceed latent capacity ({total_elements})")

        indices = self._get_indices(latent.shape)

        # Sign-nudging: preserve magnitude, adjust sign to encode bit
        for j, idx in enumerate(indices):
            bit = bits[j]
            val = flat_latent[idx]

            if bit == 1:
                # Ensure positive with margin
                if val < margin:
                    flat_latent[idx] = margin + abs(val) * 0.1
            else:
                # Ensure negative with margin
                if val > -margin:
                    flat_latent[idx] = -margin - abs(val) * 0.1

        result = flat_latent.reshape(latent.shape)
        if isinstance(latent_tensor, torch.Tensor):
            return torch.from_numpy(result).to(latent_tensor.dtype)
        return result

    def extract_from_latent(self, latent_tensor):
        if isinstance(latent_tensor, torch.Tensor):
            latent = latent_tensor.cpu().numpy()
        else:
            latent = latent_tensor

        flat_latent = latent.flatten()
        indices = self._get_indices(latent.shape)

        probs = norm.cdf(flat_latent[indices])
        extracted_bits = (probs > 0.5).astype(np.uint8)

        reshaped = extracted_bits.reshape(-1, self.repetition)
        voted_bits = (np.mean(reshaped, axis=1) > 0.5).astype(np.uint8)

        text, bitflips = self._bits_to_text(voted_bits)
        return text, bitflips

    def extract_detailed(self, latent_tensor):
        """Extract watermark with full diagnostic information."""
        if isinstance(latent_tensor, torch.Tensor):
            latent = latent_tensor.cpu().numpy()
        else:
            latent = latent_tensor

        flat_latent = latent.flatten()
        original_bits = self.encode()
        indices = self._get_indices(latent.shape)

        # 1. Raw extraction via CDF
        probs = norm.cdf(flat_latent[indices])
        raw_bits = (probs > 0.5).astype(np.uint8)

        # Raw BER (before any correction)
        ber_raw = float(np.mean(raw_bits != original_bits))

        # 2. Repetition voting
        reshaped = raw_bits.reshape(-1, self.repetition)
        vote_scores = np.mean(reshaped, axis=1)
        voted_bits = (vote_scores > 0.5).astype(np.uint8)

        # Vote margins: 1.0 = unanimous, 0.5 = split
        vote_margins = np.abs(vote_scores - 0.5) + 0.5

        # Ground truth bits after collapsing repetition
        original_collapsed = original_bits.reshape(-1, self.repetition)[:, 0]

        # BER after voting (before BCH)
        ber_voted = float(np.mean(voted_bits != original_collapsed))

        # 3. BCH decoding
        text, bitflips = self._bits_to_text(voted_bits)
        message_recovered = (text == self.wm_text)

        # BER final
        if message_recovered:
            ber_final = 0.0
        else:
            ber_final = ber_voted

        return {
            'text': text,
            'bch_corrections': bitflips,
            'raw_bits': raw_bits,
            'voted_bits': voted_bits,
            'vote_margins': vote_margins,
            'original_bits': original_bits,
            'ber_raw': float(ber_raw),
            'ber_voted': float(ber_voted),
            'ber_final': float(ber_final),
            'message_recovered': message_recovered,
            'avg_vote_margin': float(np.mean(vote_margins)),
            'min_vote_margin': float(np.min(vote_margins)),
        }


if __name__ == "__main__":
    wm_msg = "ECC_ROBUST"
    wm = ECCWatermarker(wm_text=wm_msg, bch_bits=8, repetition=5)

    bits = wm.encode()
    print(f"Original Text: {wm_msg}")
    print(f"Code Length (bits): {len(bits)}")

    # Print capacity for different image sizes
    for img_sz in [256, 512, 1024]:
        lat = img_sz // 8
        info = wm.get_capacity_info((1, 4, lat, lat))
        print(f"  {img_sz}x{img_sz}: {info['utilization_pct']:.1f}% of target channel(s), fits={info['fits']}")

    latent = wm.map_to_latent(bits)

    # Test with heavy noise
    for noise_level in [0.1, 0.3, 0.5]:
        noise = torch.randn_like(latent) * noise_level
        noisy_latent = latent + noise
        extracted_text, flips = wm.extract_from_latent(noisy_latent)
        print(f"Noise {noise_level} | Extracted: {extracted_text} | Flips: {flips}")
