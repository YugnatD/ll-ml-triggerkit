"""patch7 -> patch7: the same trigger twice, the second one fed only 32 of the 50 samples.

Level 2 is the SAME patch7 with the SAME threshold, so it should keep every level-1
trigger. With the first 32 samples (a window that ignores the trigger time) it does
not: it drops the events whose pulse comes later. Aligned on the level-1 trigger
time (what the FPGA readout does) it keeps them all.

    python examples/cascade_patch7_patch7.py "GAMMA_GLOB" "NSB_GLOB"
"""
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from triggerkit.TriggerChain import TriggerChain
from triggerkit.cascade import AlignedClamped, Fixed, Full, TriggerCascade

TARGET_RATE_HZ = 50_000
LEVEL2_SAMPLES = 32


def patch7(gamma, nsb, tau, num_samples=None):
    chain = TriggerChain(gamma, simtel_nsb_path=nsb, num_samples=num_samples)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    threshold = chain.add_stage("threshold", init_tau=tau, temp=1.0, binary_output=True)
    chain.compile_chain()
    return chain, threshold


def main():
    gamma, nsb = sorted(glob.glob(sys.argv[1])), sorted(glob.glob(sys.argv[2]))

    level1, threshold = patch7(gamma, nsb, tau=200.0)
    tau, _ = level1.find_threshold_for_target_rate(
        TARGET_RATE_HZ, nsb_roll_copies=0, nsb_skip_original_events=False)
    threshold.tau.assign(tau)
    level2, _ = patch7(gamma, nsb, tau=tau, num_samples=LEVEL2_SAMPLES)

    windows = {
        "first 32 samples": Fixed(0, LEVEL2_SAMPLES),
        "32 samples around the level-1 trigger": AlignedClamped(LEVEL2_SAMPLES, offset=LEVEL2_SAMPLES // 2),
    }
    print(f"\npatch7 threshold {tau:g} (level 1 tuned to {TARGET_RATE_HZ / 1e3:.0f} kHz)")
    for label, window in windows.items():
        cascade = TriggerCascade([level1, level2], names=["level 1", "level 2"], windows=[Full(), window])
        stats = cascade.compute_statistics()
        level1_out, level2_out = zip(stats["gamma"]["stages"], stats["nsb"]["stages"])
        print(f"\nlevel 2 on the {label}:")
        for g, n in (level1_out, level2_out):
            print(f"  {g['stage']}: gamma {100 * g['cumulative_fraction']:6.2f} %  "
                  f"NSB {n['cumulative_rate_hz'] / 1e3:5.1f} kHz")
        g, n = level2_out
        print(f"  level 2 keeps {100 * g['conditional_fraction']:.1f} % of the level-1 gammas "
              f"and {100 * n['conditional_fraction']:.1f} % of the level-1 NSB")


if __name__ == "__main__":
    main()
