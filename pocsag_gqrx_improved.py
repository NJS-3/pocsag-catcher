#!/usr/bin/env python3
"""
POCSAG Capture Analyser v2.0
For when you've got radio captures and questionable life choices.
Now with actual proper signal processing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.io import wavfile
import json
import logging
from pathlib import Path

# Import the decoder from the main file
from pocsag_improved import POCSAGDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class POCSAGAnalyser:
    """
    Analyses captured POCSAG signals from WAV files.
    Like CSI: Radio Edition, but with actual science.
    """
    
    def __init__(self, filename):
        self.filename = Path(filename)
        self.decoder = None  # Will initialise once we detect baudrate
        self.sample_rate = None
        self.signal = None
        self.is_complex = False
        
    def load_capture(self):
        """Load WAV file from GQRX with proper I/Q handling"""
        try:
            if not self.filename.exists():
                logger.error(f"File not found: {self.filename}")
                return False
            
            self.sample_rate, data = wavfile.read(self.filename)
            
            logger.info(f"📂 Loaded: {self.filename}")
            logger.info(f"   Sample rate: {self.sample_rate} Hz")
            logger.info(f"   Duration: {len(data)/self.sample_rate:.2f} seconds")
            logger.info(f"   Samples: {len(data)}")
            
            # Normalise to float
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)
            
            # Check if stereo (I/Q) or mono
            if len(data.shape) > 1 and data.shape[1] == 2:
                # Stereo - treat as I/Q
                i = data[:, 0]
                q = data[:, 1]
                self.signal = i + 1j * q  # Complex representation
                self.is_complex = True
                logger.info("   Format: Complex I/Q (stereo)")
            else:
                # Mono - real-valued signal
                self.signal = data
                self.is_complex = False
                logger.info("   Format: Real (mono)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return False
    
    def analyse_spectrum(self, max_freq=50000):
        """
        Spectral analysis with better visualisation.
        Because everyone loves a good spectrogram.
        """
        try:
            # Use magnitude if complex
            plot_signal = np.abs(self.signal) if self.is_complex else self.signal
            
            # Compute spectrogram
            f, t, Sxx = scipy_signal.spectrogram(
                plot_signal,
                self.sample_rate,
                nperseg=1024,
                noverlap=896,  # More overlap for better resolution
                window='hann'
            )
            
            # Limit frequency range to something meaningful
            freq_mask = np.abs(f) <= max_freq
            f_plot = f[freq_mask]
            Sxx_plot = Sxx[freq_mask, :]
            
            # Create figure
            fig = plt.figure(figsize=(14, 8))
            
            # Spectrogram
            ax1 = plt.subplot(3, 1, 1)
            pcm = ax1.pcolormesh(
                t, f_plot, 
                10 * np.log10(Sxx_plot + 1e-10),  # Avoid log(0)
                shading='auto', 
                cmap='viridis',
                vmin=-80,
                vmax=-20
            )
            ax1.set_ylabel('Frequency [Hz]')
            ax1.set_title('Spectrogram - The Radio Rainbow™')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(pcm, ax=ax1, label='Power [dB]')
            
            # Time domain
            ax2 = plt.subplot(3, 1, 2)
            time_axis = np.arange(min(self.sample_rate, len(plot_signal))) / self.sample_rate
            plot_len = min(self.sample_rate, len(plot_signal))
            ax2.plot(time_axis, plot_signal[:plot_len], alpha=0.7, linewidth=0.5)
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('Amplitude')
            ax2.set_title('Time Domain - The Wiggly Truth')
            ax2.grid(True, alpha=0.3)
            
            # Power spectral density
            ax3 = plt.subplot(3, 1, 3)
            f_psd, psd = scipy_signal.welch(
                plot_signal,
                self.sample_rate,
                nperseg=2048,
                window='hann'
            )
            freq_mask_psd = np.abs(f_psd) <= max_freq
            ax3.semilogy(f_psd[freq_mask_psd], psd[freq_mask_psd])
            ax3.set_xlabel('Frequency [Hz]')
            ax3.set_ylabel('PSD [V²/Hz]')
            ax3.set_title('Power Spectral Density - Where The Energy Lives')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error analysing spectrum: {e}")
    
    def detect_baudrate(self):
        """
        Auto-detect baudrate using autocorrelation.
        Actually scientific this time.
        """
        logger.info("🔍 Detecting baudrate...")
        
        # FM demodulate first
        demod = self._fm_demod(self.signal[:min(len(self.signal), self.sample_rate*2)])
        
        if len(demod) == 0:
            logger.warning("Demodulation failed, using default 1200 baud")
            return 1200
        
        # Try different baudrates
        baudrates = [512, 1200, 2400]
        scores = {}
        
        for baud in baudrates:
            samples_per_bit = int(self.sample_rate / baud)
            
            # Autocorrelation approach - look for periodicity
            # Limit correlation length for performance
            max_lag = samples_per_bit * 100
            signal_chunk = demod[:min(len(demod), max_lag * 2)]
            
            if len(signal_chunk) < samples_per_bit * 10:
                continue
            
            # Normalise signal
            signal_normalised = (signal_chunk - np.mean(signal_chunk)) / (np.std(signal_chunk) + 1e-10)
            
            # Calculate autocorrelation at expected bit period
            lag = samples_per_bit
            if lag < len(signal_normalised) // 2:
                correlation = np.correlate(
                    signal_normalised[:len(signal_normalised)//2],
                    signal_normalised[lag:lag + len(signal_normalised)//2],
                    mode='valid'
                )
                score = np.max(np.abs(correlation))
                scores[baud] = score
        
        if not scores:
            logger.warning("Could not detect baudrate, using default 1200")
            return 1200
        
        best_baud = max(scores, key=scores.get)
        confidence = scores[best_baud] / sum(scores.values()) * 100
        
        logger.info(f"\n🎯 Baudrate Detection Results:")
        for baud, score in sorted(scores.items()):
            marker = ' ← BEST' if baud == best_baud else ''
            logger.info(f"   {baud:4d} baud: {score:8.2f}{marker}")
        logger.info(f"   Confidence: {confidence:.1f}%")
        
        return best_baud
    
    def _fm_demod(self, signal):
        """
        FM demodulation optimised for captured signals.
        With proper complex signal handling this time.
        """
        try:
            if self.is_complex:
                # Complex I/Q demodulation
                # Phase difference method
                phase = np.angle(signal)
                freq = np.diff(np.unwrap(phase)) / (2 * np.pi) * self.sample_rate
            else:
                # Real signal - use Hilbert transform
                analytic = scipy_signal.hilbert(signal)
                phase = np.unwrap(np.angle(analytic))
                freq = np.diff(phase) / (2 * np.pi) * self.sample_rate
            
            # Low-pass filter
            cutoff = 5000  # Hz - above POCSAG baudrates
            nyquist = self.sample_rate / 2
            
            if cutoff < nyquist and len(freq) > 0:
                b, a = scipy_signal.butter(4, cutoff/nyquist, btype='low')
                filtered = scipy_signal.filtfilt(b, a, freq)
                return filtered
            
            return freq
            
        except Exception as e:
            logger.error(f"FM demod error: {e}")
            return np.array([])
    
    def decode_capture(self, baudrate=None):
        """
        Decode the entire capture.
        Time to unleash the bit beast.
        """
        if baudrate is None:
            baudrate = self.detect_baudrate()
        
        logger.info(f"\n🔓 Decoding at {baudrate} baud...")
        
        # Initialise decoder with detected baudrate
        self.decoder = POCSAGDecoder(baudrate=baudrate)
        
        # FM demodulate entire signal
        logger.info("Demodulating signal...")
        demod = self._fm_demod(self.signal)
        
        if len(demod) == 0:
            logger.error("Demodulation failed")
            return []
        
        # Extract bits
        logger.info("Extracting bits...")
        samples_per_bit = int(self.sample_rate / baudrate)
        bits = ""
        
        # Adaptive threshold
        threshold = np.median(demod)
        
        for i in range(0, len(demod) - samples_per_bit, samples_per_bit):
            bit_slice = demod[i:i + samples_per_bit]
            
            # Sample middle of bit period
            sample_start = len(bit_slice) // 4
            sample_end = 3 * len(bit_slice) // 4
            bit_sample = bit_slice[sample_start:sample_end]
            
            if len(bit_sample) > 0:
                avg = np.mean(bit_sample)
                bits += "1" if avg > threshold else "0"
        
        logger.info(f"Extracted {len(bits)} bits")
        
        # Decode in chunks to avoid memory issues
        logger.info("Decoding POCSAG messages...")
        chunk_size = 10000
        for i in range(0, len(bits), chunk_size):
            chunk = bits[i:i + chunk_size]
            self.decoder.decode_stream(chunk)
            
            # Progress indicator
            progress = (i / len(bits)) * 100
            if i % (chunk_size * 10) == 0:
                logger.info(f"Progress: {progress:.1f}%")
        
        # Finalise any remaining message
        if self.decoder.current_message.get('data'):
            self.decoder._finalise_message()
        
        return self.decoder.messages
    
    def save_results(self, output_file='pocsag_decoded.json'):
        """Save decoded messages to JSON"""
        try:
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(self.decoder.messages, f, indent=2)
            logger.info(f"\n💾 Results saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def print_statistics(self):
        """
        Print comprehensive analysis statistics.
        Numbers that make you look smart.
        """
        print("\n" + "="*70)
        print("📊 CAPTURE ANALYSIS STATISTICS")
        print("="*70)
        
        print(f"\nCapture Information:")
        print(f"   File: {self.filename}")
        print(f"   Duration: {len(self.signal)/self.sample_rate:.2f} seconds")
        print(f"   Sample rate: {self.sample_rate} Hz")
        print(f"   Signal type: {'Complex I/Q' if self.is_complex else 'Real'}")
        
        if self.decoder:
            print(f"\nDecoder Statistics:")
            print(f"   Baudrate: {self.decoder.baudrate}")
            print(f"   Batches processed: {self.decoder.stats['batches_processed']}")
            print(f"   Sync losses: {self.decoder.stats['sync_losses']}")
            print(f"   Errors corrected: {self.decoder.stats['corrected_errors']}")
            print(f"   Uncorrectable errors: {self.decoder.stats['uncorrectable_errors']}")
            
            messages = self.decoder.messages
            print(f"\nMessage Statistics:")
            print(f"   Total messages: {len(messages)}")
            
            if messages:
                # Address statistics
                addresses = [msg['address'] for msg in messages]
                unique_addresses = set(addresses)
                print(f"   Unique addresses: {len(unique_addresses)}")
                
                # Most common addresses
                from collections import Counter
                addr_counts = Counter(addresses)
                print(f"   Most active addresses:")
                for addr, count in addr_counts.most_common(5):
                    print(f"      {addr:6d} (0x{addr:06X}): {count} messages")
                
                # Message type breakdown
                numeric = sum(1 for msg in messages if msg['type'] == 'numeric')
                alpha = len(messages) - numeric
                print(f"\n   Message types:")
                print(f"      Numeric: {numeric} ({numeric/len(messages)*100:.1f}%)")
                print(f"      Alpha:   {alpha} ({alpha/len(messages)*100:.1f}%)")
                
                # Function codes
                from collections import Counter
                func_counts = Counter(msg['function'] for msg in messages)
                print(f"\n   Function codes:")
                for func, count in sorted(func_counts.items()):
                    print(f"      Function {func}: {count} messages")
                
                # Message lengths
                lengths = [len(msg.get('text', '')) for msg in messages]
                if lengths:
                    print(f"\n   Message length statistics:")
                    print(f"      Average: {np.mean(lengths):.1f} characters")
                    print(f"      Median:  {np.median(lengths):.0f} characters")
                    print(f"      Min:     {min(lengths)} characters")
                    print(f"      Max:     {max(lengths)} characters")
        
        print("\n" + "="*70)


def main():
    """
    Main entry point with improved argument handling.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="POCSAG Capture Analyser v2.0 - Now with less wishful thinking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s capture.wav --plot
  %(prog)s capture.wav --baudrate 1200 --output results.json
  %(prog)s capture.wav --plot --baudrate 2400
        """
    )
    
    parser.add_argument('filename', 
                       help='WAV file from GQRX or other SDR software')
    parser.add_argument('--baudrate', 
                       type=int, 
                       choices=[512, 1200, 2400],
                       help='Force specific baudrate (512/1200/2400)')
    parser.add_argument('--plot', 
                       action='store_true',
                       help='Show spectrum analysis plots')
    parser.add_argument('--output', 
                       default='pocsag_decoded.json',
                       help='Output JSON file (default: pocsag_decoded.json)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              POCSAG Capture Analyser v2.0                         ║
║         Extracting ancient secrets from RF captures              ║
║              "It's not eavesdropping, it's archaeology"           ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Create analyser
    analyser = POCSAGAnalyser(args.filename)
    
    # Load capture
    if not analyser.load_capture():
        logger.error("Failed to load capture file")
        return 1
    
    # Show spectrum if requested
    if args.plot:
        logger.info("Generating spectrum plots...")
        analyser.analyse_spectrum()
    
    # Decode
    try:
        messages = analyser.decode_capture(baudrate=args.baudrate)
        logger.info(f"\n✨ Decoded {len(messages)} messages")
    except KeyboardInterrupt:
        logger.info("\n\nDecoding interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during decoding: {e}")
        return 1
    
    # Save results
    if messages:
        analyser.save_results(args.output)
    else:
        logger.warning("No messages decoded - check frequency, baudrate, and signal quality")
    
    # Print stats
    analyser.print_statistics()
    
    logger.info("\n✨ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
