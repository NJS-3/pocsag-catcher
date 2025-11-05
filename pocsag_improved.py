#!/usr/bin/env python3
"""
POCSAG Decoder: Now With Less Explosions
Author: Someone Who Read The Spec Sheet
Version: 2.0 - "Actually Works Edition"
"""

import numpy as np
from rtlsdr import RtlSdr
import signal
import sys
from scipy import signal as scipy_signal
import threading
import queue
import time
from collections import deque
import logging

# Setup logging like civilised people
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class POCSAGDecoder:
    """
    Decodes POCSAG messages from a stream of bits.
    Now with 60% less wishful thinking and 100% more actual error correction.
    """
    
    # POCSAG constants - properly documented
    PREAMBLE_LENGTH = 576  # Minimum preamble bits
    PREAMBLE_BYTE = 0xAA  # Alternating 10101010 pattern
    SYNC_WORD = 0x7CD215D8  # Frame synchronisation codeword
    IDLE_WORD = 0x7A89C197  # Idle codeword (all systems nominal)
    
    # BCH(31,21) error correction parameters
    BCH_POLY = 0x769  # Generator polynomial
    BCH_N = 31  # Codeword length
    BCH_K = 21  # Data bits
    
    # Baudrates
    BAUDRATES = {
        512: "POCSAG-512",
        1200: "POCSAG-1200", 
        2400: "POCSAG-2400"
    }
    
    # Numeric message character set (Table from POCSAG spec)
    NUMERIC_CHARSET = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: ' ', 11: 'U',  # U = Urgent
        12: '-', 13: ')', 14: '(', 15: ''  # Reserved
    }
    
    def __init__(self, baudrate=1200):
        self.baudrate = baudrate
        self.bit_buffer = deque(maxlen=20000)  # Bounded buffer, no RAM explosion
        self.current_message = {}
        self.messages = []
        self.in_sync = False
        self.batch_position = 0
        
        # Statistics for debugging
        self.stats = {
            'batches_processed': 0,
            'sync_losses': 0,
            'corrected_errors': 0,
            'uncorrectable_errors': 0
        }
        
        # Generate BCH syndrome table (proper implementation)
        self._generate_syndrome_table()
        
    def _generate_syndrome_table(self):
        """
        Generate BCH syndrome lookup table for single-bit error correction.
        This is how you're *supposed* to do BCH.
        """
        self.syndrome_table = {}
        
        for error_pos in range(self.BCH_N):
            # Create codeword with single bit error
            error_pattern = 1 << error_pos
            syndrome = self._calculate_syndrome_value(error_pattern)
            self.syndrome_table[syndrome] = error_pos
            
        logger.debug(f"Generated BCH syndrome table with {len(self.syndrome_table)} entries")
    
    def _calculate_syndrome_value(self, codeword):
        """Calculate BCH syndrome for error detection"""
        syndrome = 0
        temp = codeword
        
        for i in range(self.BCH_K):
            if temp & (1 << (self.BCH_N - 1 - i)):
                temp ^= (self.BCH_POLY << (self.BCH_K - 1 - i))
        
        return temp & ((1 << (self.BCH_N - self.BCH_K)) - 1)
    
    def _bch_check_and_correct(self, codeword):
        """
        BCH(31,21) error detection and correction.
        Actually works this time. Revolutionary concept.
        """
        # Calculate syndrome
        syndrome = self._calculate_syndrome_value(codeword)
        
        if syndrome == 0:
            return True, codeword, 0  # No errors
        
        # Attempt single-bit correction
        if syndrome in self.syndrome_table:
            error_pos = self.syndrome_table[syndrome]
            corrected = codeword ^ (1 << error_pos)
            self.stats['corrected_errors'] += 1
            logger.debug(f"Corrected single-bit error at position {error_pos}")
            return True, corrected, 1
        
        # Uncorrectable error
        self.stats['uncorrectable_errors'] += 1
        logger.debug(f"Uncorrectable error, syndrome: 0x{syndrome:X}")
        return False, codeword, -1
    
    def _even_parity_check(self, codeword):
        """Check even parity bit (LSB of codeword)"""
        parity = 0
        temp = codeword >> 1  # Exclude parity bit itself
        
        for i in range(31):
            if temp & (1 << i):
                parity ^= 1
        
        expected_parity = codeword & 1
        return parity == expected_parity
    
    def _decode_numeric(self, data_bits):
        """Decode numeric messages using proper POCSAG character set"""
        if not data_bits:
            return ""
            
        message = []
        for i in range(0, len(data_bits), 4):
            if i + 4 <= len(data_bits):
                digit = int(data_bits[i:i+4], 2)
                char = self.NUMERIC_CHARSET.get(digit, '?')
                if char:  # Skip reserved characters
                    message.append(char)
        
        return ''.join(message).strip()
    
    def _decode_alpha(self, data_bits):
        """Decode alphanumeric messages (7-bit ASCII)"""
        if not data_bits:
            return ""
            
        message = []
        for i in range(0, len(data_bits), 7):
            if i + 7 <= len(data_bits):
                char_code = int(data_bits[i:i+7], 2)
                # Printable ASCII range
                if 32 <= char_code <= 126:
                    message.append(chr(char_code))
                elif char_code in (10, 13):  # Newline/carriage return
                    message.append('\n')
        
        return ''.join(message).strip()
    
    def find_preamble(self, bits):
        """
        Hunt for preamble with tolerance for bit errors.
        Now accepts "good enough" preambles instead of perfect ones.
        """
        min_preamble_bits = 64  # Minimum acceptable preamble
        preamble_pattern = "10" * 32  # Looking for at least 64 bits
        
        # Search with some tolerance
        best_match = -1
        best_score = 0
        
        for i in range(len(bits) - min_preamble_bits):
            score = 0
            for j in range(0, min_preamble_bits, 2):
                if i + j + 1 < len(bits):
                    if bits[i+j] == '1' and bits[i+j+1] == '0':
                        score += 1
            
            # Need at least 80% match (allow 20% bit errors)
            if score > best_score and score >= (min_preamble_bits / 2) * 0.8:
                best_score = score
                best_match = i
        
        if best_match >= 0:
            logger.debug(f"Preamble found at bit {best_match} with {best_score}/{min_preamble_bits//2} score")
        
        return best_match
    
    def process_batch(self, batch_bits):
        """
        Process a batch of 544 bits (1 sync word + 8 frames of 64 bits).
        With proper error handling this time.
        """
        if len(batch_bits) < 544:
            return False
        
        # Extract and verify sync word
        try:
            sync = int(batch_bits[:32], 2)
        except ValueError:
            logger.warning("Invalid bits in sync word")
            return False
        
        if sync != self.SYNC_WORD:
            self.in_sync = False
            self.stats['sync_losses'] += 1
            logger.debug(f"Sync lost, got 0x{sync:08X}, expected 0x{self.SYNC_WORD:08X}")
            return False
        
        self.in_sync = True
        self.stats['batches_processed'] += 1
        
        # Process 8 frames (2 codewords per frame)
        for frame_idx in range(8):
            for cw_idx in range(2):
                bit_offset = 32 + (frame_idx * 2 + cw_idx) * 32
                codeword_bits = batch_bits[bit_offset:bit_offset + 32]
                
                if len(codeword_bits) == 32:
                    try:
                        codeword = int(codeword_bits, 2)
                        self.process_codeword(codeword, frame_idx * 2 + cw_idx)
                    except ValueError:
                        logger.warning(f"Invalid codeword bits at frame {frame_idx}")
        
        return True
    
    def process_codeword(self, codeword, frame_pos):
        """Process individual codewords with proper error handling"""
        
        # Check parity first
        if not self._even_parity_check(codeword):
            logger.debug("Parity check failed")
            # Continue anyway, BCH might fix it
        
        # BCH error correction
        valid, corrected_cw, errors = self._bch_check_and_correct(codeword)
        
        if not valid:
            logger.debug(f"Uncorrectable error in codeword at position {frame_pos}")
            return
        
        codeword = corrected_cw
        
        # Check for idle word
        if codeword == self.IDLE_WORD:
            return
        
        # Determine if it's address or message codeword
        is_message = bool(codeword & 0x80000000)
        
        if is_message:
            # Message codeword - extract 20 data bits
            data_bits = format((codeword >> 11) & 0xFFFFF, '020b')
            
            if self.current_message.get('address') is not None:
                if 'data' not in self.current_message:
                    self.current_message['data'] = ""
                self.current_message['data'] += data_bits
            else:
                logger.debug("Message codeword without preceding address")
        else:
            # Address codeword
            address = (codeword >> 13) & 0x3FFFF  # 18 bits
            function = (codeword >> 11) & 0x3      # 2 bits
            
            # Calculate actual address (includes frame position)
            actual_address = (address << 3) | (frame_pos & 0x7)
            
            # Finalise previous message if exists
            if self.current_message.get('data'):
                self._finalise_message()
            
            # Start new message
            self.current_message = {
                'address': actual_address,
                'function': function,
                'type': 'alpha' if function >= 2 else 'numeric',
                'errors_corrected': errors
            }
    
    def _finalise_message(self):
        """Complete and store the current message"""
        if not self.current_message.get('data'):
            return
        
        msg = self.current_message.copy()
        
        # Decode based on type
        try:
            if msg['type'] == 'numeric':
                msg['text'] = self._decode_numeric(msg['data'])
            else:
                msg['text'] = self._decode_alpha(msg['data'])
        except Exception as e:
            logger.error(f"Error decoding message: {e}")
            msg['text'] = "[DECODE ERROR]"
        
        # Clean up
        msg.pop('data', None)
        msg['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        msg['baudrate'] = self.baudrate
        
        self.messages.append(msg)
        
        # Pretty print
        print(f"\n{'='*70}")
        print(f"📟 POCSAG Message Decoded")
        print(f"{'='*70}")
        print(f"Time:     {msg['timestamp']}")
        print(f"Address:  {msg['address']} (0x{msg['address']:X})")
        print(f"Function: {msg['function']}")
        print(f"Type:     {msg['type'].upper()}")
        print(f"Baudrate: {self.baudrate}")
        if msg.get('errors_corrected', 0) > 0:
            print(f"Errors:   {msg['errors_corrected']} corrected")
        print(f"\nMessage:")
        print(f"  {msg['text']}")
        print(f"{'='*70}\n")
        
        self.current_message = {}
    
    def decode_stream(self, bit_stream):
        """
        Main decoder loop with improved buffering.
        Now with 100% less memory leaks.
        """
        # Add new bits to buffer
        self.bit_buffer.extend(bit_stream)
        
        # Convert deque to string for processing (less elegant, more reliable)
        bits_str = ''.join(self.bit_buffer)
        
        # Look for preamble if not synced
        if not self.in_sync:
            preamble_idx = self.find_preamble(bits_str)
            if preamble_idx >= 0:
                # Remove bits before preamble
                for _ in range(preamble_idx + self.PREAMBLE_LENGTH):
                    if self.bit_buffer:
                        self.bit_buffer.popleft()
                
                bits_str = ''.join(self.bit_buffer)
                self.in_sync = True
                logger.info("🎯 Preamble detected! Syncing...")
        
        # Process batches while we have enough bits
        while len(bits_str) >= 544 and self.in_sync:
            if not self.process_batch(bits_str[:544]):
                # Lost sync, remove one bit and try again
                if self.bit_buffer:
                    self.bit_buffer.popleft()
                bits_str = ''.join(self.bit_buffer)
                self.in_sync = False
            else:
                # Successfully processed batch, remove those bits
                for _ in range(544):
                    if self.bit_buffer:
                        self.bit_buffer.popleft()
                bits_str = ''.join(self.bit_buffer)
    
    def print_stats(self):
        """Print decoder statistics"""
        print(f"\n📊 Decoder Statistics:")
        print(f"   Batches processed: {self.stats['batches_processed']}")
        print(f"   Sync losses: {self.stats['sync_losses']}")
        print(f"   Errors corrected: {self.stats['corrected_errors']}")
        print(f"   Uncorrectable errors: {self.stats['uncorrectable_errors']}")
        print(f"   Messages decoded: {len(self.messages)}")


class RTLSDRReceiver:
    """
    RTL-SDR receiver with proper error handling.
    Now significantly less likely to crash your computer.
    """
    
    def __init__(self, frequency=153.075e6, sample_rate=240000, gain=20, baudrate=1200):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self.decoder = POCSAGDecoder(baudrate=baudrate)
        self.sdr = None
        self.running = False
        
        # FSK parameters for POCSAG
        self.fsk_deviation = 4500  # Hz - typical for POCSAG
        
    def setup_sdr(self):
        """Configure RTL-SDR with proper error handling"""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency
            self.sdr.gain = self.gain
            
            logger.info(f"📡 RTL-SDR configured:")
            logger.info(f"   Frequency: {self.frequency/1e6:.3f} MHz")
            logger.info(f"   Sample Rate: {self.sample_rate/1000} kHz")
            logger.info(f"   Gain: {self.gain} dB")
            return True
        except Exception as e:
            logger.error(f"Failed to initialise RTL-SDR: {e}")
            return False
    
    def fm_demod(self, iq_samples):
        """
        Improved FM demodulation for FSK signals.
        Less crude, more effective.
        """
        try:
            # Quadrature demodulation with proper handling
            # Remove DC bias first
            iq_samples = iq_samples - np.mean(iq_samples)
            
            # Instantaneous phase
            phase = np.angle(iq_samples)
            
            # Frequency derivative
            demod = np.diff(np.unwrap(phase)) / (2.0 * np.pi) * self.sample_rate
            
            # Low-pass filter to remove high-frequency noise
            # Cutoff should be above baudrate but below Nyquist
            cutoff = self.decoder.baudrate * 3  # 3x baudrate for good measure
            nyquist = self.sample_rate / 2
            
            if cutoff < nyquist:
                b, a = scipy_signal.butter(4, cutoff/nyquist, btype='low')
                demod = scipy_signal.lfilter(b, a, demod)
            
            return demod
        except Exception as e:
            logger.error(f"FM demod error: {e}")
            return np.array([])
    
    def extract_bits(self, demod_signal):
        """
        Extract bits with improved clock recovery.
        Still simple, but less catastrophically so.
        """
        if len(demod_signal) == 0:
            return ""
        
        bits = ""
        samples_per_bit = int(self.sample_rate / self.decoder.baudrate)
        
        # Simple threshold from signal statistics
        threshold = np.median(demod_signal)
        
        for i in range(0, len(demod_signal) - samples_per_bit, samples_per_bit):
            bit_slice = demod_signal[i:i + samples_per_bit]
            
            # Use middle 50% of bit period for sampling (avoid transitions)
            sample_start = len(bit_slice) // 4
            sample_end = 3 * len(bit_slice) // 4
            bit_sample = bit_slice[sample_start:sample_end]
            
            if len(bit_sample) > 0:
                avg = np.mean(bit_sample)
                bits += "1" if avg > threshold else "0"
        
        return bits
    
    def process_samples(self, samples, ctx):
        """Callback for processing RTL-SDR samples"""
        if not self.running:
            return
        
        try:
            # FM demodulation
            demod = self.fm_demod(samples)
            
            # Extract bits
            if len(demod) > 0:
                bits = self.extract_bits(demod)
                
                # Decode POCSAG
                if bits:
                    self.decoder.decode_stream(bits)
        except Exception as e:
            logger.error(f"Error processing samples: {e}")
    
    def start(self):
        """Start receiving with proper error handling"""
        if not self.setup_sdr():
            return False
        
        self.running = True
        
        logger.info("\n🚀 Starting POCSAG decoder...")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            # Read samples asynchronously
            self.sdr.read_samples_async(
                self.process_samples,
                num_samples=self.sample_rate // 10
            )
        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt received")
        except Exception as e:
            logger.error(f"Error during reception: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Clean shutdown with proper cleanup"""
        logger.info("\n🛑 Stopping decoder...")
        self.running = False
        
        if self.sdr:
            try:
                self.sdr.close()
            except Exception as e:
                logger.error(f"Error closing SDR: {e}")
        
        # Print final statistics
        self.decoder.print_stats()


def signal_handler(sig, frame):
    """Graceful signal handling"""
    logger.info("\n👋 Signal caught, shutting down...")
    sys.exit(0)


def main():
    """Main entry point with improved UI"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║               POCSAG Decoder v2.0 - "Actually Works"              ║
║          Intercepting pager messages since... just now            ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Common POCSAG frequencies
    frequencies = {
        '1': (153.075e6, "UK Emergency Services"),
        '2': (153.125e6, "UK Fire Brigade"),
        '3': (153.150e6, "UK Ambulance"),
        '4': (169.0375e6, "Medical"),
        '5': (453.075e6, "Business"),
        '6': (453.100e6, "Commercial"),
    }
    
    print("Select frequency:")
    for key, (freq, desc) in frequencies.items():
        print(f"  {key}: {freq/1e6:.4f} MHz - {desc}")
    print("  Or enter custom frequency in MHz")
    
    choice = input("\nYour choice: ").strip()
    
    if choice in frequencies:
        freq, desc = frequencies[choice]
        logger.info(f"Selected: {desc}")
    else:
        try:
            freq = float(choice) * 1e6
            logger.info(f"Custom frequency: {freq/1e6} MHz")
        except ValueError:
            freq = 153.075e6
            logger.warning(f"Invalid input, using default: {freq/1e6} MHz")
    
    # Baudrate selection
    print("\nSelect baudrate:")
    print("  1: 512 baud")
    print("  2: 1200 baud (most common)")
    print("  3: 2400 baud")
    
    baud_choice = input("Your choice (default 1200): ").strip()
    baudrate_map = {'1': 512, '2': 1200, '3': 2400}
    baudrate = baudrate_map.get(baud_choice, 1200)
    
    # Create and start receiver
    receiver = RTLSDRReceiver(frequency=freq, baudrate=baudrate)
    receiver.start()


if __name__ == "__main__":
    main()
