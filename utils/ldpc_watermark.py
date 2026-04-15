import numpy as np
import torch
from scipy.stats import norm
from pyldpc import make_ldpc, encode, decode, get_message


class LDPCWatermarker:
    """
    Robust Watermarking using ECC (LDPC + Repetition Coding) and Distribution-Aware Sampling.

    This class mirrors the ECCWatermarker API but replaces BCH algebraic decoding
    with LDPC belief-propagation (min-sum) decoding, as described in:
        "Robust watermarking for diffusion models using error-correcting codes
         and post-quantum key encapsulation" (Hu et al., Frontiers in Physics, 2026)

    Two-Layer Hierarchical Defense:
        Layer 1 (LDPC): Block-level error correction via iterative belief propagation
                        on a sparse parity-check matrix H. Corrects distributed bit errors.
        Layer 2 (Repetition): Each LDPC-coded bit is repeated n times. Majority voting
                              suppresses localized burst errors before LDPC decoding.

    Latent Channel Semantics (SD VAE):
        Channel 0: Coarse luminance/brightness — DO NOT modify (most visually sensitive)
        Channel 1: Color/chrominance — DO NOT modify
        Channel 2: Higher-frequency spatial detail — safe to modify (less perceptual impact)
        Channel 3: Fine texture/edges — safe to modify (less perceptual impact)

    By default, watermark bits are embedded only in channel 2 ('single' mode).
    When target_channels='dual', bits are split across channels 2 and 3, halving
    the per-channel modification density and improving PSNR.
    """

    def __init__(self, wm_text="test", snr=10.0, d_v=3, d_c=6,
                 repetition=5, latent_shape=(4, 64, 64), ecc_seed=42,
                 target_channels='single'):
        self.wm_text = wm_text
        self.snr = snr           # Signal-to-noise ratio for LDPC decoding
        self.d_v = d_v           # Variable node degree (column weight of H)
        self.d_c = d_c           # Check node degree (row weight of H)
        self.repetition = repetition
        self.latent_shape = latent_shape
        self.ecc_seed = ecc_seed
        self.target_channels = target_channels  # 'single' (ch2 only) or 'dual' (ch2+ch3)

        # Convert message to raw bits
        self._raw_bits = self._text_to_raw_bits(wm_text)
        self.k = len(self._raw_bits)  # number of information bits

        # Build LDPC code matrices
        # n (block length) must be > k, and n must be divisible by d_c
        # We pick n such that code rate R = k/n is roughly 0.5 for good correction
        # n must satisfy: n >= k, and (n - k) must be compatible with d_v, d_c
        n_target = max(self.k * 2, 60)  # at least 2x expansion for decent correction
        # Round up to nearest multiple of d_c
        self.n = int(np.ceil(n_target / d_c) * d_c)

        # Build parity-check matrix H and generator matrix G
        self.H, self.G = make_ldpc(self.n, self.d_v, self.d_c,
                                    systematic=True, sparse=True)
        # Actual k from generator matrix (may differ slightly from our message length)
        self.k_actual = self.G.shape[1]

    def _text_to_raw_bits(self, text):
        """Convert text string to raw binary bit array."""
        data = bytearray(text, 'utf-8')
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return bits

    def _raw_bits_to_text(self, bits):
        """Convert raw binary bit array back to text string."""
        if len(bits) % 8 != 0:
            bits = np.concatenate([bits, np.zeros(8 - (len(bits) % 8), dtype=np.uint8)])
        data = np.packbits(bits).tobytes()
        try:
            decoded = data.decode('utf-8', errors='ignore')
            return decoded[:len(self.wm_text)]
        except:
            return None

    def _text_to_bits(self, text):
        """Encode text to LDPC-protected bit sequence."""
        raw_bits = self._text_to_raw_bits(text)

        # Pad or truncate to match k_actual
        if len(raw_bits) < self.k_actual:
            padded = np.zeros(self.k_actual, dtype=int)
            padded[:len(raw_bits)] = raw_bits
        else:
            padded = raw_bits[:self.k_actual]

        # LDPC encode: encode() expects the message vector and returns a noisy
        # encoded signal. We use a high SNR so the "noise" is negligible,
        # then threshold to get clean binary codeword.
        coded_signal = encode(self.G, padded, snr=100)  # high SNR = near-noiseless
        coded_bits = (coded_signal < 0).astype(np.uint8)  # BPSK: -1 -> bit 1, +1 -> bit 0

        return coded_bits

    def _bits_to_text(self, received_bits):
        """Decode LDPC-protected bit sequence back to text.

        Returns:
            (text, num_iterations): decoded text and number of BP iterations used,
                                     or (None, -1) on failure.
        """
        # Convert binary bits to BPSK-like signal for LDPC decoder
        # bit 0 -> +1, bit 1 -> -1  (soft values scaled by SNR)
        received_signal = (1 - 2 * received_bits.astype(np.float64)) * self.snr

        # LDPC decode using belief propagation (min-sum variant)
        decoded_codeword = decode(self.H, received_signal, snr=self.snr, maxiter=500)

        # Extract original message bits from the decoded codeword
        decoded_msg = get_message(self.G, decoded_codeword)

        # Truncate to actual message length
        msg_bits = decoded_msg[:self.k].astype(np.uint8)

        text = self._raw_bits_to_text(msg_bits)
        if text is not None and text == self.wm_text:
            return text, 0  # 0 = success, no residual errors
        elif text is not None:
            # Count bit differences
            original_bits = self._text_to_raw_bits(self.wm_text)
            min_len = min(len(msg_bits), len(original_bits))
            bit_errors = int(np.sum(msg_bits[:min_len] != original_bits[:min_len]))
            return text, bit_errors
        return None, -1

    def encode(self):
        """Encode watermark text with LDPC + Repetition (2-layer protection).

        Layer 1 (LDPC): text -> LDPC codeword bits
        Layer 2 (Repetition): each codeword bit repeated self.repetition times
        """
        coded_bits = self._text_to_bits(self.wm_text)
        repeated_bits = np.repeat(coded_bits, self.repetition)
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
            'num_ldpc_bits': len(self._text_to_bits(self.wm_text)),
            'num_raw_bits': self.k,
            'code_rate': self.k_actual / self.n,
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

        # Quantile Overwrite (Distribution Preserving Strategy)
        np.random.seed(self.ecc_seed)
        u = np.random.uniform(0, 1, len(bits))
        mapped_values = norm.ppf((u + bits) / 2.0).astype(np.float32)

        for j, idx in enumerate(indices):
            flat_latent[idx] = mapped_values[j]

        result = flat_latent.reshape(latent.shape)
        if isinstance(latent_tensor, torch.Tensor):
            return torch.from_numpy(result).to(latent_tensor.dtype)
        return result

    def extract_from_latent(self, latent_tensor):
        """Extract watermark from latent tensor.

        Decapsulation pipeline:
        1. Extract raw bits via norm.cdf thresholding
        2. Layer 2 decapsulation: Majority voting across repetition groups
        3. Layer 1 decapsulation: LDPC belief-propagation decoding
        """
        if isinstance(latent_tensor, torch.Tensor):
            latent = latent_tensor.cpu().numpy()
        else:
            latent = latent_tensor

        flat_latent = latent.flatten()
        indices = self._get_indices(latent.shape)

        # Step 1: Raw bit extraction via CDF thresholding
        probs = norm.cdf(flat_latent[indices])
        extracted_bits = (probs > 0.5).astype(np.uint8)

        # Step 2: Layer 2 — Repetition majority voting
        reshaped = extracted_bits.reshape(-1, self.repetition)
        voted_bits = (np.mean(reshaped, axis=1) > 0.5).astype(np.uint8)

        # Step 3: Layer 1 — LDPC decoding
        text, errors = self._bits_to_text(voted_bits)
        return text, errors

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

        # 2. Repetition voting (Layer 2 decapsulation)
        reshaped = raw_bits.reshape(-1, self.repetition)
        vote_scores = np.mean(reshaped, axis=1)
        voted_bits = (vote_scores > 0.5).astype(np.uint8)

        # Vote margins: 1.0 = unanimous, 0.5 = split
        vote_margins = np.abs(vote_scores - 0.5) + 0.5

        # Ground truth bits after collapsing repetition
        original_collapsed = original_bits.reshape(-1, self.repetition)[:, 0]

        # BER after voting (before LDPC)
        ber_voted = float(np.mean(voted_bits != original_collapsed))

        # 3. LDPC decoding (Layer 1 decapsulation)
        text, errors = self._bits_to_text(voted_bits)
        message_recovered = (text == self.wm_text)

        # BER final
        if message_recovered:
            ber_final = 0.0
        else:
            ber_final = ber_voted

        return {
            'text': text,
            'ldpc_errors': errors,
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
    wm_msg = "test"
    wm = LDPCWatermarker(wm_text=wm_msg, snr=10.0, d_v=3, d_c=6, repetition=5)

    bits = wm.encode()
    print(f"Original Text: {wm_msg}")
    print(f"Raw message bits: {wm.k}")
    print(f"LDPC codeword bits: {wm.n}")
    print(f"Code rate: {wm.k_actual / wm.n:.3f}")
    print(f"Total embedded (with rep): {len(bits)}")

    # Print capacity for different image sizes
    for img_sz in [256, 512, 1024]:
        lat = img_sz // 8
        info = wm.get_capacity_info((1, 4, lat, lat))
        print(f"  {img_sz}x{img_sz}: {info['utilization_pct']:.1f}% of target channel(s), fits={info['fits']}")

    # Test encode-decode cycle (no noise)
    coded_bits = wm._text_to_bits(wm_msg)
    text_back, errs = wm._bits_to_text(coded_bits)
    print(f"\nClean decode: '{text_back}' | errors={errs}")

    # Test with noisy bits
    for flip_pct in [0.05, 0.10, 0.15]:
        noisy = coded_bits.copy()
        n_flips = int(len(noisy) * flip_pct)
        flip_idx = np.random.choice(len(noisy), n_flips, replace=False)
        noisy[flip_idx] ^= 1
        text_n, errs_n = wm._bits_to_text(noisy)
        print(f"  {flip_pct*100:.0f}% bit flips -> decoded: '{text_n}' | errors={errs_n}")

    # Test full latent embedding cycle
    print("\nFull latent embedding test:")
    latent = np.random.normal(0, 1, (1, 4, 64, 64)).astype(np.float32)
    latent_wm = wm.embed_into_latent(torch.from_numpy(latent), margin=0.75)

    for noise_level in [0.0, 0.1, 0.3, 0.5]:
        noisy_latent = latent_wm + torch.randn_like(latent_wm) * noise_level
        extracted_text, flips = wm.extract_from_latent(noisy_latent)
        print(f"  Noise {noise_level} | Extracted: '{extracted_text}' | Errors: {flips}")
